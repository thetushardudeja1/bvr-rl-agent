"""
================================================================================
 RUN_tacview.py  --  PRESS THE ▶ "RUN" BUTTON IN VS CODE TO GENERATE A REPLAY
================================================================================
No terminal typing needed. Just open this file in VS Code and click Run
(the ▶ triangle, top-right) or press F5.

It plays the trained Blue agent (RL) against the Red behaviour-tree, finds an
engagement where Blue beams/notches Red's missiles and wins, and writes a
Tacview .acmi replay you can open in Tacview.

Output  ->  ./output_tacview/   (inside this project folder)

You can change WHAT it generates with the CONFIG block just below.
================================================================================
"""
import os, sys, shutil
try:                                    # Windows cp1252 can't print emoji/arrows
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# --- make the project importable no matter where VS Code launches from -------
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                 # tacview output dir is relative -> anchor it here
sys.path.insert(0, HERE)

# ============================== CONFIG ========================================
MODEL      = 'experiments/rir/rir_notch2_best'   # the trained Blue agent
HOW_MANY   = 3            # how many qualifying replays to save
SEED_START = 7500         # first random seed to try
SEED_END   = 9500         # last seed to try
# A replay is kept only if: Blue WINS, Red fired, Blue beat >=1 missile by the
# NOTCH, and Blue used ZERO run-away (drag) defeats -> pure beam-and-win.
REQUIRE_PURE_NOTCH = True
# ==============================================================================

OUT = os.path.join(HERE, 'output_tacview')
os.makedirs(OUT, exist_ok=True)

from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents
from jsb_gym.utils.loggers import TacviewLogger

print("Loading trained agent ...")
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

found, seed = 0, SEED_START
print(f"Scanning seeds {SEED_START}..{SEED_END} for winning beam-and-notch fights ...")
while found < HOW_MANY and seed <= SEED_END:
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)
    bsh = rsh = 0; pb = pr = False; min_d = 1e9
    seen = {}; n_notch = n_kin = 0
    done = trunc = False
    while not (done or trunc):
        env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        b, r = env.blue_agent, env.red_agent
        min_d = min(min_d, dinstance_between_agents(b, r))
        ba, ra = b.is_own_missile_active(), r.is_own_missile_active()
        bsh += (ba and not pb); rsh += (ra and not pr); pb, pr = ba, ra
        for m in r.ammo.values():
            if (m.target_lost or getattr(m, 'coasting', False)) and id(m) not in seen:
                if m.notch_count > m.missile_performance.notch_count:
                    seen[id(m)] = 'n'; n_notch += 1
                elif m.target_lost:
                    seen[id(m)] = 'k'; n_kin += 1
    oc = info.get('outcome', '?')
    keep = (oc == 'win' and rsh > 0 and
            (not REQUIRE_PURE_NOTCH or (n_notch > 0 and n_kin == 0)))
    if keep:
        for _ in range(12):                       # post-kill coast for a clean trail
            env.tacview_logger.log_flight_data()
            a, _ = model.predict(obs, deterministic=True)
            try: obs, _, _, _, info = env.step(a)
            except Exception: break
        env.tacview_logger.save_logs()
        found += 1
        name = f"replay_seed{seed}_B{bsh}R{rsh}_beam{n_notch}_drag{n_kin}_min{min_d/1e3:.0f}km.acmi"
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(OUT, name))
        print(f"  SAVED ({found}/{HOW_MANY}): {name}")
    seed += 1

print(f"\nDONE. {found} replay(s) written to:\n  {OUT}")
print("Open the .acmi files in Tacview to watch them.")
