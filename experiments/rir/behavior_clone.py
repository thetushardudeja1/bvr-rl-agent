"""
behavior_clone.py
=================
Stage 2 of RIR — "Imitation". Supervised behavior cloning of the BT expert
into a PPO MlpPolicy, producing a warm-started model. Also saves the policy
state_dict to use as the FROZEN reference for the AEG loss in stage 3.

Usage:
    python -m experiments.rir.behavior_clone \
        --data experiments/rir/bt_dataset.npz \
        --epochs 30 --out experiments/rir/bc_model
"""
import sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '.')

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='experiments/rir/bt_dataset.npz')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', type=str, default='experiments/rir/bc_model')
    p.add_argument('--val_frac', type=float, default=0.1)
    args = p.parse_args()

    d = np.load(args.data)
    obs = torch.as_tensor(d['obs'], dtype=torch.float32)
    acts = torch.as_tensor(d['acts'], dtype=torch.float32)
    n = len(obs)
    idx = torch.randperm(n)
    obs, acts = obs[idx], acts[idx]
    n_val = int(n * args.val_frac)
    tr_obs, tr_act = obs[n_val:], acts[n_val:]
    va_obs, va_act = obs[:n_val], acts[:n_val]
    print(f"  dataset: {n:,} pairs  (train {len(tr_obs):,} / val {len(va_obs):,})")

    # single dummy env just to build the policy with correct spaces
    env = make_vec_env(BVRBase, n_envs=1, env_kwargs={'conf': baseEnv_conf})
    model = PPO("MlpPolicy", env, device='cpu', verbose=0)
    policy = model.policy
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    def policy_mean(obs_b):
        feats = policy.extract_features(obs_b)
        if policy.share_features_extractor:
            latent_pi, _ = policy.mlp_extractor(feats)
        else:
            pi_feats, _ = feats
            latent_pi = policy.mlp_extractor.forward_actor(pi_feats)
        return policy.action_net(latent_pi)

    nb = max(1, len(tr_obs) // args.batch)
    for ep in range(args.epochs):
        perm = torch.randperm(len(tr_obs))
        tot = 0.0
        for b in range(nb):
            bi = perm[b*args.batch:(b+1)*args.batch]
            pred = policy_mean(tr_obs[bi])
            loss = F.mse_loss(pred, tr_act[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        with torch.no_grad():
            vpred = policy_mean(va_obs)
            vloss = F.mse_loss(vpred, va_act).item()
            vmae = (vpred - va_act).abs().mean().item()
        print(f"  epoch {ep+1:>3}/{args.epochs}  train_mse={tot/nb:.5f}  "
              f"val_mse={vloss:.5f}  val_mae={vmae:.4f}")

    model.save(args.out)
    torch.save(policy.state_dict(), args.out + "_policy.pt")
    print(f"\n  ✅ BC model saved -> {args.out}.zip")
    print(f"     frozen reference -> {args.out}_policy.pt")
    env.close()


if __name__ == "__main__":
    main()
