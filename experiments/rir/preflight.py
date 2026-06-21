"""
preflight.py
============
Pre-training sanity checks — answers "will this actually learn?" WITHOUT
running the full job. Three checks:

  1. GRADIENT/MACHINERY HEALTH  — PPO+AEG runs a few updates: losses finite,
     no NaN in params, policy weights actually move.
  2. REWARD REACHABILITY        — with the warm-started policy, do wins/kills
     actually OCCUR? (If every episode is a -50 timeout there is no win signal
     to learn from, no matter how long you train.)
  3. WARM-START vs RANDOM       — does the BC policy beat a random policy?
     (Confirms the warm-start is meaningful, not noise.)

Usage:
    python -m experiments.rir.preflight --bc experiments/rir/bc_model
"""
import sys, argparse, copy
import numpy as np
import torch
sys.path.insert(0, '.')

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.envs.vec_env import ChunkSubprocVecEnv
from experiments.rir.ppo_aeg import PPO_AEG


def classify(ret):
    if ret > 50:    return 'WIN'
    if ret < -100:  return 'LOSS'
    return 'TIMEOUT'


def reachability(predict_fn, n_envs=30, max_steps=2600):
    env = make_vec_env(BVRBase, n_envs=n_envs, vec_env_cls=ChunkSubprocVecEnv,
                       env_kwargs={'conf': baseEnv_conf})
    obs = env.reset()
    ep_ret = np.zeros(n_envs)
    rets = []
    for _ in range(max_steps):
        action = predict_fn(obs, n_envs, env)
        obs, rew, done, infos = env.step(action)
        ep_ret += rew
        for i in range(n_envs):
            if done[i]:
                rets.append(ep_ret[i]); ep_ret[i] = 0.0
    env.close()
    return rets


def gradient_health(bc_path):
    env = make_vec_env(BVRBase, n_envs=24, vec_env_cls=ChunkSubprocVecEnv,
                       env_kwargs={'conf': baseEnv_conf})
    m = PPO_AEG("MlpPolicy", env, device='cpu', n_steps=64, batch_size=512,
                gamma=0.999, ent_coef=0.01, aeg_coef_start=1.0, aeg_coef_end=0.0,
                verbose=0)
    bc = PPO.load(bc_path, device='cpu')
    m.policy.load_state_dict(bc.policy.state_dict())
    m.reference_policy = copy.deepcopy(m.policy)
    for p in m.reference_policy.parameters(): p.requires_grad_(False)
    m.reference_policy.eval()

    before = torch.cat([p.detach().flatten() for p in m.policy.parameters()]).clone()
    m.learn(total_timesteps=64 * 24 * 4)
    after = torch.cat([p.detach().flatten() for p in m.policy.parameters()])
    delta = (after - before).norm().item()
    any_nan = bool(torch.isnan(after).any())
    ev = m.logger.name_to_value.get('train/explained_variance', float('nan'))
    aeg = m.logger.name_to_value.get('train/aeg_loss', float('nan'))
    env.close()
    return delta, any_nan, ev, aeg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bc', type=str, default='experiments/rir/bc_model')
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("  PRE-FLIGHT CHECK 1: GRADIENT / MACHINERY HEALTH")
    print("=" * 60)
    delta, any_nan, ev, aeg = gradient_health(args.bc)
    print(f"  policy weight movement (L2) : {delta:.4f}   (>0 = learning)")
    print(f"  NaN in params               : {any_nan}")
    print(f"  explained_variance          : {ev:.3f}")
    print(f"  AEG loss computed           : {aeg:.4f}")
    c1 = (delta > 1e-4) and (not any_nan)
    print(f"  -> {'✅ PASS' if c1 else '❌ FAIL'}")

    print("\n" + "=" * 60)
    print("  PRE-FLIGHT CHECK 2+3: REWARD REACHABILITY")
    print("=" * 60)
    bc = PPO.load(args.bc, device='cpu')
    def bc_predict(obs, n, env):
        a, _ = bc.predict(obs, deterministic=False); return a
    def rand_predict(obs, n, env):
        return np.stack([env.action_space.sample() for _ in range(n)])

    bc_rets = reachability(bc_predict)
    rd_rets = reachability(rand_predict)

    def report(name, rets):
        from collections import Counter
        c = Counter(classify(r) for r in rets)
        n = len(rets)
        w = c.get('WIN', 0)
        print(f"  {name:>12}: {n} eps | WIN {w} ({100*w/max(1,n):.0f}%)  "
              f"LOSS {c.get('LOSS',0)}  TIMEOUT {c.get('TIMEOUT',0)}  "
              f"| mean_ret {np.mean(rets):.1f}")
        return w, n

    bw, bn = report("warm-start", bc_rets)
    rw, rn = report("random", rd_rets)

    c2 = bw > 0  # wins are reachable
    c3 = (bw / max(1, bn)) >= (rw / max(1, rn))  # warm-start >= random
    print(f"\n  Check 2 (wins reachable) : {'✅ PASS' if c2 else '❌ FAIL — no win signal!'}")
    print(f"  Check 3 (warm >= random) : {'✅ PASS' if c3 else '⚠️  warm-start not beating random'}")

    print("\n" + "=" * 60)
    verdict = c1 and c2
    if verdict:
        print("  ✅ PRE-FLIGHT PASSED — training machinery sound and")
        print("     win reward is reachable. Safe to launch.")
    else:
        print("  ❌ PRE-FLIGHT FAILED — fix before the 50M run:")
        if not c1: print("     - gradient/machinery problem")
        if not c2: print("     - NO WINS REACHABLE: reward too sparse, agent")
        print("       cannot learn to win. Need denser shaping / easier start.")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    main()
