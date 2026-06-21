"""Functional check of the V2 self-play env after EMA removal / deterministic
opponents / 4-dim actions: steps cleanly vs BT and vs a frozen policy."""
import sys
sys.path.insert(0, '.')
import numpy as np
from jsb_gym.envs.config import baseEnv_conf
from experiments.v2.selfplay_env_v2 import SelfPlayBVR

for opp in ['BT', 'experiments/rir/bc_model_v3']:
    baseEnv_conf.opponent_path = opp
    env = SelfPlayBVR(baseEnv_conf)
    obs, _ = env.reset(seed=7)
    assert obs.shape == tuple(baseEnv_conf.observation_shape)
    for i in range(15):
        a = np.random.uniform(-1, 1, 4).astype(np.float32)
        obs, r, done, trunc, info = env.step(a)
        assert np.isfinite(obs).all() and np.isfinite(r)
        if done:
            obs, _ = env.reset()
    print(f"  opponent={opp.split('/')[-1]:14s} 15 steps OK")
print("✅ SELF-PLAY ENV TEST PASSED")
