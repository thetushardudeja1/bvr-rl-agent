"""
measure_baseline.py
==================
Honest measurement of the BC warm-start's TRUE win rate over many episodes
(the earlier 39% was a small-sample fluke). Reports win/loss/timeout with a
binomial 95% CI so we know the real number to beat.
"""
import sys, argparse, math
import numpy as np
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.envs.vec_env import ChunkSubprocVecEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='experiments/rir/bc_model')
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--n_envs', type=int, default=40)
    p.add_argument('--deterministic', action='store_true')
    args = p.parse_args()

    env = make_vec_env(BVRBase, n_envs=args.n_envs, vec_env_cls=ChunkSubprocVecEnv,
                       env_kwargs={'conf': baseEnv_conf})
    model = PPO.load(args.model, device='cpu')
    obs = env.reset()
    counts = {'win': 0, 'loss': 0, 'timeout': 0}
    done_eps = 0
    while done_eps < args.episodes:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, dones, infos = env.step(action)
        for i in range(args.n_envs):
            if dones[i]:
                oc = infos[i].get('outcome', '')
                if oc in counts:
                    counts[oc] += 1
                    done_eps += 1

    n = sum(counts.values())
    w = counts['win']
    p_hat = w / n
    ci = 1.96 * math.sqrt(p_hat * (1 - p_hat) / n)
    print("\n" + "=" * 55)
    print(f"  TRUE BASELINE — {args.model}  ({'deterministic' if args.deterministic else 'stochastic'})")
    print("=" * 55)
    print(f"  episodes : {n}")
    print(f"  WIN      : {counts['win']:>4}  ({100*counts['win']/n:.1f}%)  +/- {100*ci:.1f}%")
    print(f"  LOSS     : {counts['loss']:>4}  ({100*counts['loss']/n:.1f}%)")
    print(f"  TIMEOUT  : {counts['timeout']:>4}  ({100*counts['timeout']/n:.1f}%)")
    print("=" * 55)
    env.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    main()
