"""
train_selfplay.py  (anti-collapse PFSP)
=======================================
Plain PFSP self-play collapsed (win-rate vs BT 47% -> 16%): the agent
over-specialized against recent selves and FORGOT general skill. This version
adds the documented anti-collapse mechanisms:

  1. ANCHORED sampling      — a fixed fraction of generations train against the
                              competent baselines (BT + seed) so the agent can
                              never forget basic competence (anti-forgetting).
  2. GATE                    — a new self is added to the pool only if it was
                              actually competent (win-rate > threshold), so the
                              pool never fills with collapsed junk (AOS PSW-GF).
  3. POOL CAP               — keep only the most recent N self-snapshots; old
                              ones are dropped (AOS fixed-length pool).
  4. BEST-BY-BT eval + save — clean deterministic eval vs BT each generation;
                              save the model whenever it sets a new BT high.
  5. LOWER LR               — slows drift.
"""
import sys, os, time, argparse, random
import numpy as np
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.envs.vec_env import ChunkSubprocVecEnv
from experiments.v2.selfplay_env_v2 import SelfPlayBVR
from experiments.rir.ppo_aeg import PPO_AEG
from experiments.rir.callbacks import RIRLogger


def eval_vs_bt(model, n_envs=40, episodes=60):
    """Clean deterministic win-rate vs the BT (the yardstick)."""
    baseEnv_conf.opponent_path = 'BT'
    env = make_vec_env(SelfPlayBVR, n_envs=n_envs, vec_env_cls=ChunkSubprocVecEnv,
                       env_kwargs={'conf': baseEnv_conf})
    obs = env.reset()
    wins = done = 0
    while done < episodes:
        # DETERMINISTIC: the mean policy is the real one (measured 76% det
        # vs 14% stochastic) — gating on sampled rollouts would be gating
        # on exploration noise.
        a, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(a)
        for i in range(n_envs):
            if dones[i]:
                oc = infos[i].get('outcome', '')
                if oc in ('win', 'loss', 'timeout'):
                    done += 1
                    if oc == 'win':
                        wins += 1
    env.close()
    return wins / max(1, done)


def sample_opponent(self_pool, winrate, baselines, p_baseline=0.5):
    # ANCHORED: half the time train vs a competent baseline (anti-forgetting)
    if (random.random() < p_baseline) or (not self_pool):
        return random.choice(baselines)
    # else PFSP among recent selves: prefer ones we beat less
    weights = [max(0.1, 1.0 - winrate.get(p, 0.5)) for p in self_pool]
    s = sum(weights); r = random.random() * s; c = 0.0
    for p, w in zip(self_pool, weights):
        c += w
        if r <= c:
            return p
    return self_pool[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed_model', type=str, default='experiments/rir/seed_clean')
    p.add_argument('--gen_steps', type=int, default=2_000_000)
    p.add_argument('--generations', type=int, default=38)
    p.add_argument('--n_envs', type=int, default=300)
    p.add_argument('--n_steps', type=int, default=128)
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--lr', type=float, default=5e-5)        # FIX: lower (was 1e-4)
    p.add_argument('--ent_coef', type=float, default=0.003)
    p.add_argument('--gamma', type=float, default=0.999)
    p.add_argument('--p_baseline', type=float, default=0.5) # FIX: anti-forgetting
    p.add_argument('--gate', type=float, default=0.45)      # FIX: pool quality gate
    p.add_argument('--pool_cap', type=int, default=8)       # FIX: fixed-length pool
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--outdir', type=str, default='experiments/rir/selfplay')
    # exploration control for the launch dim: fire/no-fire is effectively
    # discrete; sigma=1 white noise on it pulls the trigger ~16% of steps and
    # sabotages rollouts (measured: 76% det vs 14% stochastic). Init its
    # log_std low so weapon exploration is rare, deliberate trials.
    p.add_argument('--launch_log_std', type=float, default=-2.0)
    # CONSTANT demonstration anchor on the policy during self-play (AlphaStar
    # keeps an imitation loss throughout league training). League v2 ran with
    # bc_coef=0 and the main agent collapsed to 0% within 4 generations once
    # wins vanished from its rollouts — anchored OPPONENTS don't anchor the
    # POLICY. Small constant coef = can still innovate, can't fall off.
    p.add_argument('--bc_anchor', type=float, default=0.1)
    p.add_argument('--bc_data', type=str, default='experiments/rir/bt_dataset_v3.npz')
    # AlphaStar-style main exploiter: every K generations, train a fresh
    # seed-clone purely against the current best and add it to the pool —
    # manufactures headroom a fixed BT can't expose. 0 disables.
    p.add_argument('--exploiter_every', type=int, default=5)
    p.add_argument('--exploiter_steps', type=int, default=2_000_000)
    args = p.parse_args()

    run_name = args.run_name or f"SP_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join('runs', run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    baselines = ['BT', args.seed_model]   # permanent competent anchors
    self_pool = []                        # capped list of self-snapshots
    winrate = {}

    baseEnv_conf.opponent_path = 'BT'
    vec_env = make_vec_env(SelfPlayBVR, n_envs=args.n_envs,
                           vec_env_cls=ChunkSubprocVecEnv,
                           env_kwargs={'conf': baseEnv_conf})
    model = PPO_AEG("MlpPolicy", vec_env, device=args.device, verbose=0,
                    learning_rate=args.lr, n_steps=args.n_steps,
                    batch_size=args.batch, ent_coef=args.ent_coef,
                    gamma=args.gamma,
                    bc_coef_start=args.bc_anchor, bc_coef_end=args.bc_anchor)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
    seed = PPO.load(args.seed_model, device=args.device)
    model.policy.load_state_dict(seed.policy.state_dict())
    if args.bc_anchor > 0.0:
        d = np.load(args.bc_data)
        model.set_demonstrations(d['obs'], d['acts'])
    if args.launch_log_std is not None and model.policy.log_std.shape[0] > 3:
        import torch
        with torch.no_grad():
            model.policy.log_std[3] = args.launch_log_std
        print(f"  launch-dim exploration: log_std[3] = {args.launch_log_std} "
              f"(sigma = {float(np.exp(args.launch_log_std)):.3f})")
    vec_env.close()

    best_bt = eval_vs_bt(model)
    print(f"  seeded from {args.seed_model}; baseline win-rate vs BT = {best_bt:.0%}")
    model.save(os.path.join(args.outdir, 'selfplay_best'))

    for gen in range(args.generations):
        # ---- EXPLOITER GENERATION (AlphaStar main-exploiter pattern) ----
        # every K gens: a FRESH seed clone trains purely against the current
        # best, then joins the opponent pool. It exists to find the champion's
        # habits; it is reset to the seed each time for fresh attack angles.
        if (args.exploiter_every > 0 and gen > 0
                and gen % args.exploiter_every == 0):
            best_path = os.path.join(args.outdir, 'selfplay_best')
            print(f"\n=== GEN {gen} | EXPLOITER vs selfplay_best ===")
            baseEnv_conf.opponent_path = best_path
            exp_env = make_vec_env(SelfPlayBVR, n_envs=args.n_envs,
                                   vec_env_cls=ChunkSubprocVecEnv,
                                   env_kwargs={'conf': baseEnv_conf})
            exp_model = PPO_AEG("MlpPolicy", exp_env, device=args.device,
                                verbose=0, learning_rate=args.lr,
                                n_steps=args.n_steps, batch_size=args.batch,
                                ent_coef=args.ent_coef, gamma=args.gamma,
                                bc_coef_start=args.bc_anchor,
                                bc_coef_end=args.bc_anchor)
            exp_model.policy.load_state_dict(
                PPO.load(args.seed_model, device=args.device).policy.state_dict())
            if args.bc_anchor > 0.0:
                d = np.load(args.bc_data)
                exp_model.set_demonstrations(d['obs'], d['acts'])
            if args.launch_log_std is not None and exp_model.policy.log_std.shape[0] > 3:
                import torch
                with torch.no_grad():
                    exp_model.policy.log_std[3] = args.launch_log_std
            exp_cb = RIRLogger(save_path=os.path.join(args.outdir, f'exploiter{gen}'),
                               window=100, print_freq=50)
            exp_model.learn(total_timesteps=args.exploiter_steps,
                            callback=[exp_cb], reset_num_timesteps=True)
            exp_env.close()
            exp_snap = os.path.join(args.outdir, f'exploiter{gen}')
            exp_model.save(exp_snap)
            n = len(exp_cb.results)
            exp_wr = exp_cb.results.count('WIN') / n if n else 0.0
            # exploiters join the pool unconditionally — their job is to
            # attack the main agent, not to be BT-competent
            self_pool.append(exp_snap)
            winrate[exp_snap] = 0.5
            if len(self_pool) > args.pool_cap:
                self_pool.pop(0)
            del exp_model
            print(f"  EXPLOITER done: rollout wr vs best = {exp_wr:.0%}; added to pool")
            continue

        opp = sample_opponent(self_pool, winrate, baselines, args.p_baseline)
        tag = opp.split('/')[-1]
        print(f"\n=== GEN {gen} | opponent={tag} | selves={len(self_pool)} ===")
        baseEnv_conf.opponent_path = opp
        vec_env = make_vec_env(SelfPlayBVR, n_envs=args.n_envs,
                               vec_env_cls=ChunkSubprocVecEnv,
                               env_kwargs={'conf': baseEnv_conf})
        model.set_env(vec_env)
        logger_cb = RIRLogger(save_path=os.path.join(args.outdir, f'gen{gen}'),
                              window=100, print_freq=20)
        model.learn(total_timesteps=args.gen_steps, callback=[logger_cb],
                    reset_num_timesteps=True, progress_bar=False)
        n = len(logger_cb.results)
        wr = logger_cb.results.count('WIN') / n if n else 0.5
        winrate[opp] = wr
        vec_env.close()

        # BEST-BY-BT: clean deterministic yardstick eval
        snap = os.path.join(args.outdir, f'gen{gen}')
        model.save(snap)
        bt_wr = eval_vs_bt(model)

        # GATE on the deterministic BT eval, NOT the stochastic rollout
        # win-rate (rollouts run with exploration noise and sit far below
        # the mean policy's true strength — gating on them never passes).
        if bt_wr >= args.gate:
            self_pool.append(snap)
            winrate[snap] = 0.5
            if len(self_pool) > args.pool_cap:        # POOL CAP: drop oldest self
                self_pool.pop(0)

        flag = ''
        if bt_wr > best_bt:
            best_bt = bt_wr
            model.save(os.path.join(args.outdir, 'selfplay_best'))
            flag = ' *NEW BEST*'
        print(f"  GEN {gen} done: vs {tag}={wr:.0%} | EVAL vs BT={bt_wr:.0%} "
              f"(best {best_bt:.0%}){flag}")

    print(f"\n  ✅ self-play complete. best-vs-BT model -> {args.outdir}/selfplay_best.zip")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    main()
