"""
verify_tactics.py — characterise the retrained (smart-BT) agent's behaviour and
save a watchable Tacview of a win (post-kill coast). Shows whether it fights
(varied, decisive) rather than runs.
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
saved_win = False
print(f"{'ep':>3} | {'outcome':8s} | {'dur':>5} | mean|bank| | reversals | alt band | closest | shots B/R")
print('-'*92)
for ep in range(8):
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=3000 + ep)
    log = (not saved_win)
    if log: env.tacview_logger = TacviewLogger(env)
    phis, alts = [], []; min_d = 1e9; bsh = rsh = 0; pb = pr = False
    done = trunc = False
    while not (done or trunc):
        if log: env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        b, r = env.blue_agent, env.red_agent
        phis.append(b.simObj.get_phi()); alts.append(b.simObj.get_altitude())
        min_d = min(min_d, dinstance_between_agents(b, r))
        ba, ra = b.is_own_missile_active(), r.is_own_missile_active()
        bsh += (ba and not pb); rsh += (ra and not pr); pb, pr = ba, ra
    oc = info.get('outcome', '?')
    phi = np.array(phis); s = np.sign(phi)
    rev = int(np.sum((s[1:] != s[:-1]) & (np.abs(phi[1:]) > 15) & (np.abs(phi[:-1]) > 15)))
    print(f"{ep:>3} | {oc:8s} | {len(phis)*0.5/60:4.1f}m | {np.mean(np.abs(phi)):8.1f} "
          f"| {rev:8d} | {min(alts)/1e3:4.1f}-{max(alts)/1e3:4.1f}km | {min_d/1e3:5.1f}km | {bsh}/{rsh}")
    if log and oc == 'win':
        for _ in range(12):  # post-kill coast
            env.tacview_logger.log_flight_data()
            a, _ = model.predict(obs, deterministic=True)
            try: obs, _, _, _, info = env.step(a)
            except Exception: break
        env.tacview_logger.save_logs()
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, 'smart_agent_win.acmi'))
        saved_win = True
        print(f"    -> saved smart_agent_win.acmi")
print("\ndone")
