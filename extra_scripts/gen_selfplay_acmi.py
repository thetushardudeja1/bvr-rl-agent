"""
gen_selfplay_acmi.py — render self-play (agent-vs-agent) engagements to ACMI and
print a quantitative behaviour readout, to inspect tactical variety.
Runs in the 15-dim worktree; models referenced by absolute path.
"""
import sys, os, shutil, math
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from experiments.v2.selfplay_env_v2 import SelfPlayBVR
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents
from jsb_gym.utils.loggers import TacviewLogger

R = '/home/tushar/bvr_v2_clean/experiments/rir'
PAIRS = [
    (f'{R}/rir_model_v3_best', f'{R}/league3/gen2',         'champion_vs_gen2'),
    (f'{R}/league3/gen2',      f'{R}/league3/gen11',        'gen2_vs_gen11'),
    (f'{R}/league3/exploiter10', f'{R}/rir_model_v3_best',  'exploiter_vs_champion'),
    (f'{R}/rir_model_v3_best', f'{R}/rir_model_v3_best',    'champion_mirror'),
]
OUT = 'selfplay_acmi'; os.makedirs(OUT, exist_ok=True)
WOUT = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/proof_flight'

def run(bluep, redp, name, seed=42):
    baseEnv_conf.opponent_path = redp
    env = SelfPlayBVR(baseEnv_conf)
    blue = PPO.load(bluep, device='cpu')
    obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)          # explicit logger
    phis, alts = [], []; min_d = 1e9; bshot = rshot = 0; pb = pr = False
    done = trunc = False
    while not (done or trunc):
        env.tacview_logger.log_flight_data()
        a, _ = blue.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        b, r = env.blue_agent, env.red_agent
        phis.append(b.simObj.get_phi()); alts.append(b.simObj.get_altitude())
        min_d = min(min_d, dinstance_between_agents(b, r))
        ba, ra = b.is_own_missile_active(), r.is_own_missile_active()
        if ba and not pb: bshot += 1
        if ra and not pr: rshot += 1
        pb, pr = ba, ra
    # post-kill coast (~6 s) so the destruction is visible in the replay:
    # the killed jet stays removed, the survivor flies on over empty sky.
    for _ in range(12):
        env.tacview_logger.log_flight_data()
        a, _ = blue.predict(obs, deterministic=True)
        try:
            obs, _, _, _, info = env.step(a)
        except Exception:
            break
    env.tacview_logger.save_logs()
    src = os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi')
    dst = os.path.join(OUT, name + '.acmi')
    shutil.copy(src, dst); shutil.copy(src, os.path.join(WOUT, name + '.acmi'))
    phi = np.array(phis); sign = np.sign(phi)
    rev = int(np.sum((sign[1:] != sign[:-1]) & (np.abs(phi[1:]) > 15) & (np.abs(phi[:-1]) > 15)))
    print(f"{name:24s} | {info.get('outcome','?'):7s} "
          f"| dur {len(phis)*0.5/60:4.1f}min | mean|bank| {np.mean(np.abs(phi)):4.1f} "
          f"| reversals {rev:3d} | alt {min(alts)/1e3:4.1f}-{max(alts)/1e3:4.1f}km "
          f"| closest {min_d/1e3:5.1f}km | shots B{bshot}/R{rshot}")

print(f"{'pair':24s} | outcome | duration | mean|bank| | reversals | alt band | closest | shots")
print('-'*120)
for bp, rp, nm in PAIRS:
    try: run(bp, rp, nm)
    except Exception as e: print(f"{nm}: FAILED {e}")
print(f"\nACMIs in {OUT}/ and copied to proof_flight/")
