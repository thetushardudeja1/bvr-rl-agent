"""Sanity test for SelfPlayBVR: Blue(47%) vs Red(47% frozen). Confirms red
is flown by the frozen policy, episodes resolve, outcomes/HP behave."""
import sys
import numpy as np
from collections import Counter
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.config import baseEnv_conf
from experiments.rir.selfplay_env import SelfPlayBVR
from jsb_gym.utils.geospatial import dinstance_between_agents

baseEnv_conf.opponent_path = 'experiments/rir/rir_model_best'
blue = PPO.load('experiments/rir/rir_model_best', device='cpu')

outcomes = []
for ep in range(8):
    env = SelfPlayBVR(baseEnv_conf)
    obs, _ = env.reset()
    done = trunc = False
    steps = 0
    min_d = 1e9
    red_headings = []
    while not (done or trunc):
        a, _ = blue.predict(obs, deterministic=False)
        obs, r, done, trunc, info = env.step(a)
        min_d = min(min_d, dinstance_between_agents(env.blue_agent, env.red_agent))
        red_headings.append(env.red_agent.simObj.get_psi())
        steps += 1
    outcomes.append(info['outcome'])
    red_moved = np.std(red_headings) > 1.0   # did red actually maneuver?
    print(f"ep{ep}: {info['outcome']:>8} steps={steps:>4} min_d={min_d/1000:>4.0f}km "
          f"blueHP={env.blue_agent.healthPoints:.0f} redHP={env.red_agent.healthPoints:.0f} "
          f"red_maneuvered={red_moved}")

print("\n", Counter(outcomes))
print("Self-play (same policy both sides) should be ~symmetric win/loss with timeouts.")
