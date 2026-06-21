"""
benchmark_sps.py
================
Measures raw COLLECTION SPS (env steps/sec) for different vec-env configs.
Steps the vec env directly with random actions — no PPO, no model, no rollout
boundary issues. This is the pure throughput of the environment pipeline,
matching the methodology in SPEEDUP.md.

Does NOT save any models or touch the trained/ folder.

Usage:
    cd ~/bvrgym_project
    python benchmark_sps.py
"""

import os
import sys
import time
import multiprocessing
import numpy as np
sys.path.insert(0, '.')

from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.envs.vec_env import ChunkSubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Each config: (label, n_envs, vec_env_cls, n_iters)
# total env-steps measured = n_envs * n_iters
CONFIGS = [
    ("ChunkSubproc",  240,  ChunkSubprocVecEnv,  80),
    ("ChunkSubproc",  360,  ChunkSubprocVecEnv,  60),
]
WARMUP_ITERS = 10   # discard initial iterations (process warmup / first resets)


def run_benchmark():
    results = []

    print("\n" + "=" * 70)
    print("  Collection SPS Benchmark — BVR Gym (scalar-math + ChunkSubprocVecEnv)")
    print("=" * 70)
    print(f"  {'VecEnv':>16}  {'N Envs':>8}  {'Steps':>10}  {'SPS':>10}  {'Est 15M':>10}")
    print("-" * 70)

    for label, n_envs, vec_cls, n_iters in CONFIGS:
        vec_env = make_vec_env(
            BVRBase,
            n_envs      = n_envs,
            vec_env_cls = vec_cls,
            env_kwargs  = {'conf': baseEnv_conf},
        )

        act_dim = vec_env.action_space.shape[0]
        vec_env.reset()

        # warmup
        for _ in range(WARMUP_ITERS):
            actions = np.random.uniform(-1, 1, size=(n_envs, act_dim)).astype(np.float32)
            vec_env.step(actions)

        # timed
        start = time.time()
        for _ in range(n_iters):
            actions = np.random.uniform(-1, 1, size=(n_envs, act_dim)).astype(np.float32)
            vec_env.step(actions)
        elapsed = time.time() - start

        total_steps = n_envs * n_iters
        sps         = int(total_steps / elapsed)
        est_hours   = (15_000_000 / sps) / 3600

        results.append((label, n_envs, sps, est_hours))
        print(f"  {label:>16}  {n_envs:>8}  {total_steps:>10,}  {sps:>10,}  {est_hours:>8.1f}h")

        vec_env.close()

    print("=" * 70)
    best = max(results, key=lambda x: x[2])
    print(f"\n  Best: {best[0]} {best[1]} envs → {best[2]:,} SPS  (~{best[3]:.1f} hrs for 15M steps)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import platform
    if platform.system() != "Windows":
        multiprocessing.set_start_method("fork", force=True)
    run_benchmark()
