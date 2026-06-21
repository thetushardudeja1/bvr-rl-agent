"""
ppo_aeg.py
==========
PPO with DAPG-style auxiliary Behavior-Cloning loss (Rajeswaran et al. 2017).

Each policy update adds an auxiliary term that regresses the policy toward the
ACTUAL expert demonstration actions (the BT dataset), with a decaying weight:

    L = L_ppo + bc_coef(t) * MSE( pi_mean(demo_obs), demo_actions )

This is the proven fix for BC->RL policy collapse / catastrophic forgetting:
unlike a KL-to-frozen-reference (which is ~0 early and weak), the BC loss is a
constant, strong anchor to the real demonstrations throughout fine-tuning.
bc_coef decays start->end so demonstrations guide early exploration but don't
bias the final policy.
"""
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class PPO_AEG(PPO):
    def __init__(self, *args,
                 bc_coef_start=1.0, bc_coef_end=0.1, bc_batch_size=2048,
                 # legacy KL-AEG args kept for API compat (unused by default)
                 reference_policy=None, aeg_coef_start=0.0, aeg_coef_end=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_coef_start = bc_coef_start
        self.bc_coef_end   = bc_coef_end
        self.bc_batch_size = bc_batch_size
        self.bc_dims   = None       # if set (e.g. [0,1,2]), anchor only these
                                    # action dims — leaves others free for RL
        self.demo_obs  = None
        self.demo_acts = None
        # scheduling (set by the two-phase trainer)
        self._bc_active = True     # disabled during value-only warmup
        self._bc_t0 = 0            # num_timesteps at start of phase 2
        self._bc_total = None      # phase-2 horizon for the decay
        # legacy (unused)
        self.reference_policy = reference_policy

    def set_demonstrations(self, obs, acts):
        self.demo_obs  = torch.as_tensor(obs,  dtype=torch.float32, device=self.device)
        self.demo_acts = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        print(f"  DAPG demonstrations loaded: {len(self.demo_obs):,} pairs")

    def _excluded_save_params(self):
        # NEVER pickle the demo tensors: ~1.5GB serialized into every
        # checkpoint/best save (2GB zips) — the save-time memory spike was
        # OOM-killing hour-long training runs. Demos are re-attached by the
        # trainer via set_demonstrations() after load.
        return super()._excluded_save_params() + ["demo_obs", "demo_acts"]

    def _current_bc_coef(self):
        if (not self._bc_active) or (self.demo_obs is None):
            return 0.0
        total = self._bc_total or self._total_timesteps
        elapsed = self.num_timesteps - self._bc_t0
        frac = 1.0 - (elapsed / max(1, total))
        frac = min(1.0, max(0.0, frac))
        return self.bc_coef_end + frac * (self.bc_coef_start - self.bc_coef_end)

    def _policy_mean(self, obs):
        feats = self.policy.extract_features(obs)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(feats)
        else:
            pi_feats, _ = feats
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_feats)
        return self.policy.action_net(latent_pi)

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        bc_coef = self._current_bc_coef()
        pg_losses, value_losses, ent_losses, bc_losses = [], [], [], []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                p1 = advantages * ratio
                p2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(p1, p2).mean()
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is not None:
                    values = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                # ---- DAPG auxiliary BC loss on real demonstrations ----
                bc_loss = torch.tensor(0.0, device=self.device)
                if bc_coef > 0.0 and self.demo_obs is not None:
                    n = self.demo_obs.shape[0]
                    idx = torch.randint(0, n, (min(self.bc_batch_size, n),),
                                        device=self.device)
                    pred = self._policy_mean(self.demo_obs[idx])
                    tgt = self.demo_acts[idx]
                    if self.bc_dims is not None:   # anchor only selected dims
                        pred = pred[:, self.bc_dims]
                        tgt = tgt[:, self.bc_dims]
                    bc_loss = F.mse_loss(pred, tgt)

                loss = (policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + bc_coef * bc_loss)

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                ent_losses.append(entropy_loss.item())
                bc_losses.append(float(bc_loss.item()))

        self._n_updates += self.n_epochs
        ev = explained_variance(self.rollout_buffer.values.flatten(),
                                self.rollout_buffer.returns.flatten())
        self.logger.record("train/policy_loss", sum(pg_losses)/len(pg_losses))
        self.logger.record("train/value_loss", sum(value_losses)/len(value_losses))
        self.logger.record("train/bc_loss", sum(bc_losses)/len(bc_losses))
        self.logger.record("train/bc_coef", bc_coef)
        self.logger.record("train/explained_variance", ev)
