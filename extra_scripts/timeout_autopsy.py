"""
timeout_autopsy.py — hunt timeout episodes of the seed vs BT and characterize
them: starting geometry, altitude profiles, who ever fired, closest approach.
Decides whether timeouts are unwinnable geometry (tiny WEZ) or passivity.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents

model = PPO.load('experiments/rir/bc_model_v3', device='cpu')

found = 0
seed_i = 5000
while found < 5 and seed_i < 5200:
    env = BVRBase(baseEnv_conf)
    obs, _ = env.reset(seed=seed_i)
    d0 = dinstance_between_agents(env.blue_agent, env.red_agent)
    b_alt0 = env.blue_agent.simObj.get_altitude()
    r_alt0 = env.red_agent.simObj.get_altitude()
    min_d, b_fired, r_fired = float('inf'), 0, 0
    alts = []
    prev_b = prev_r = False
    done = trunc = False
    while not (done or trunc):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(act)
        d = dinstance_between_agents(env.blue_agent, env.red_agent)
        min_d = min(min_d, d)
        alts.append(env.blue_agent.simObj.get_altitude())
        b = env.blue_agent.is_own_missile_active()
        r = env.red_agent.is_own_missile_active()
        if b and not prev_b: b_fired += 1
        if r and not prev_r: r_fired += 1
        prev_b, prev_r = b, r
    if info.get('outcome') == 'timeout':
        found += 1
        print(f"TIMEOUT seed={seed_i}: start R={d0/1e3:5.1f}km "
              f"alts B{b_alt0/1e3:.1f}/R{r_alt0/1e3:.1f}km | min_R={min_d/1e3:5.1f}km "
              f"| blue_alt mean={np.mean(alts)/1e3:.1f} end={alts[-1]/1e3:.1f}km "
              f"| shots B={b_fired} R={r_fired}")
    seed_i += 1

print(f"\nscanned {seed_i-5000} episodes to find {found} timeouts")
