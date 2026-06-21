"""Instrument BC-vs-BT episodes: who fires first, from what range/alt/aspect,
and how it correlates with outcome. Distinguishes 'BC is just weak' from
'structural launch asymmetry'."""
import sys
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents

model = PPO.load('experiments/rir/bc_model_v2', device='cpu')

N_EP = 10
for ep in range(N_EP):
    env = BVRBase(baseEnv_conf)
    obs, _ = env.reset(seed=2000 + ep)
    blue_first, red_first = None, None
    blue_n, red_n = 0, 0
    prev_b, prev_r = False, False
    done = trunc = False
    while not (done or trunc):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(act)
        t = env.blue_agent.simObj.get_sim_time_sec()
        b = env.blue_agent.is_own_missile_active()
        r = env.red_agent.is_own_missile_active()
        if b and not prev_b:
            blue_n += 1
            if blue_first is None:
                blue_first = (t, dinstance_between_agents(env.blue_agent, env.red_agent) / 1e3,
                              env.blue_agent.simObj.get_altitude() / 1e3)
        if r and not prev_r:
            red_n += 1
            if red_first is None:
                red_first = (t, dinstance_between_agents(env.red_agent, env.blue_agent) / 1e3,
                             env.red_agent.simObj.get_altitude() / 1e3)
        prev_b, prev_r = b, r

    fmt = lambda x: f"t={x[0]:5.0f}s R={x[1]:5.1f}km alt={x[2]:4.1f}km" if x else "NEVER FIRED"
    print(f"ep{ep}: {info.get('outcome','?'):8s} | blue 1st: {fmt(blue_first)} (n={blue_n}) "
          f"| red 1st: {fmt(red_first)} (n={red_n})")
