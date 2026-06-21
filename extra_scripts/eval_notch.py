"""
eval_notch.py — deterministic eval of the notch-trained agent vs the STANDARD
smart red (doctrine off, scale 1.0), and — the point of this whole exercise —
classify HOW blue defeats each incoming red missile:
   notch     : red missile dropped lock via the Doppler notch (blue BEAMED)
   kinematic : red missile lost via opening-distance / low energy (blue DRAGGED)
This tells us whether blue developed a second evasion or still only runs.
"""
import sys, argparse, numpy as np
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents

p = argparse.ArgumentParser()
p.add_argument('--model', default='experiments/rir/rir_notch')
p.add_argument('--eps', type=int, default=40)
args = p.parse_args()

baseEnv_conf.red_doctrine_randomize = False     # standard smart red for eval
model = PPO.load(args.model, device='cpu')

wins = 0; losses = 0; timeouts = 0
ep_notch = 0; ep_drag = 0; ep_both = 0
tot_notch = 0; tot_kinematic = 0
for s in range(6000, 6000 + args.eps):
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=s)
    # remember per-missile state to detect the moment/cause of loss
    seen = {}                       # id(missile) -> classified
    used_notch = used_drag = False
    done = trunc = False
    while not (done or trunc):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        for m in env.red_agent.ammo.values():
            if m.target_lost and id(m) not in seen:
                lim = m.missile_performance.notch_count
                if m.notch_count > lim:
                    seen[id(m)] = 'notch'; used_notch = True; tot_notch += 1
                else:
                    seen[id(m)] = 'kin';   used_drag = True; tot_kinematic += 1
    oc = info.get('outcome', '?')
    wins += oc == 'win'; losses += oc == 'loss'; timeouts += oc == 'timeout'
    if used_notch and used_drag: ep_both += 1
    elif used_notch: ep_notch += 1
    elif used_drag:  ep_drag += 1

n = args.eps
print(f"\n=== {args.model}  ({n} eps, deterministic, standard smart red) ===")
print(f"  win {wins}/{n} = {wins/n:.0%}   loss {losses}   timeout {timeouts}")
print(f"\n  RED MISSILES DEFEATED, by cause:")
print(f"    notch (blue BEAMED)      : {tot_notch}")
print(f"    kinematic (blue DRAGGED) : {tot_kinematic}")
print(f"\n  EPISODES by evasion repertoire blue used:")
print(f"    beamed only      : {ep_notch}")
print(f"    dragged only     : {ep_drag}")
print(f"    BOTH (varied)    : {ep_both}")
print(f"    -> blue used the notch in {ep_notch+ep_both}/{n} episodes "
      f"({(ep_notch+ep_both)/n:.0%})")
