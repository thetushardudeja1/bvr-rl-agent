"""
gen_mutual_win.py — find and save Tacview replays where BOTH sides fire missiles
AND blue wins (a genuine two-sided duel won by the agent). Bank-to-turn missile,
post-kill coast so the kill is visible.
"""
import sys, os, shutil
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents
from jsb_gym.utils.loggers import TacviewLogger

MODEL = 'experiments/rir/checkpoints/rir_14011680_steps'
WOUT = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/proof_flight'
model = PPO.load(MODEL, device='cpu')
want, found, seed = 3, 0, 4000
print("scanning for blue-wins where both sides fire...")
while found < want and seed < 4080:
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)
    bsh = rsh = 0; pb = pr = False; min_d = 1e9
    done = trunc = False
    while not (done or trunc):
        env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        b, r = env.blue_agent, env.red_agent
        min_d = min(min_d, dinstance_between_agents(b, r))
        ba, ra = b.is_own_missile_active(), r.is_own_missile_active()
        bsh += (ba and not pb); rsh += (ra and not pr); pb, pr = ba, ra
    oc = info.get('outcome', '?')
    if oc == 'win' and bsh > 0 and rsh > 0:
        for _ in range(12):  # post-kill coast
            env.tacview_logger.log_flight_data()
            a, _ = model.predict(obs, deterministic=True)
            try: obs, _, _, _, info = env.step(a)
            except Exception: break
        env.tacview_logger.save_logs()
        found += 1
        name = f"mutual_win{found}_seed{seed}_B{bsh}R{rsh}_min{min_d/1e3:.0f}km.acmi"
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, name))
        print(f"  saved {name}")
    else:
        print(f"  seed {seed}: {oc} B{bsh}/R{rsh} -- skip")
    seed += 1
print(f"\n{found} mutual-fire blue-win replays saved to proof_flight/")
