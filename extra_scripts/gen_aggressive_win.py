"""
gen_aggressive_win.py — SELECTOR (does NOT modify sim/agent/missile). It plays
episodes with the unmodified 14M model and SAVES only the ones where blue does
the opposite of the turn-and-run habit:
  (1) outcome == win,
  (2) BOTH sides fire >=1 missile (bsh>0 and rsh>0),
  (3) blue does NOT extend after red's first shot: after red launches, blue KEEPS
      CLOSING (min range afterwards < CLOSE_FRAC * range-at-red-launch) and stays
      mostly nose-on to red (median aspect < ASPECT_MAX deg).
Bank-to-turn missile, post-kill coast so the kill is visible.
"""
import sys, os, shutil, math
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents
from jsb_gym.utils.loggers import TacviewLogger

MODEL      = 'experiments/rir/checkpoints/rir_14011680_steps'
WOUT       = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/two_sided_duels'
CLOSE_FRAC = 0.75     # blue must close to <75% of the range it had when red fired
ASPECT_MAX = 75.0     # blue must stay nose-on (median aspect-to-red below this)
WANT       = 4
os.makedirs(WOUT, exist_ok=True)
model = PPO.load(MODEL, device='cpu')

def aspect_deg(b, r):
    # angle between blue's nose (psi) and the bearing blue->red; 0 = pointing at red
    blat, blon = math.radians(b.simObj.get_lat_gc_deg()), math.radians(b.simObj.get_long_gc_deg())
    rlat, rlon = math.radians(r.simObj.get_lat_gc_deg()), math.radians(r.simObj.get_long_gc_deg())
    dlon = rlon - blon
    y = math.sin(dlon) * math.cos(rlat)
    x = math.cos(blat)*math.sin(rlat) - math.sin(blat)*math.cos(rlat)*math.cos(dlon)
    brg = math.degrees(math.atan2(y, x)) % 360
    psi = b.simObj.get_psi() % 360
    return abs((brg - psi + 180) % 360 - 180)

found, seed = 0, 5000
print("scanning for AGGRESSIVE blue-wins (both fire, blue keeps closing)...")
while found < WANT and seed < 5200:
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)
    bsh = rsh = 0; pb = pr = False; min_d = 1e9
    d_at_rfire = None; min_d_after = 1e9; aspects_after = []
    done = trunc = False
    while not (done or trunc):
        env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        b, r = env.blue_agent, env.red_agent
        d = dinstance_between_agents(b, r); min_d = min(min_d, d)
        ba, ra = b.is_own_missile_active(), r.is_own_missile_active()
        bsh += (ba and not pb)
        red_fires_now = (ra and not pr)
        rsh += red_fires_now
        if red_fires_now and d_at_rfire is None:
            d_at_rfire = d                       # range the moment red first shoots
        if d_at_rfire is not None:
            min_d_after = min(min_d_after, d)
            aspects_after.append(aspect_deg(b, r))
        pb, pr = ba, ra
    oc = info.get('outcome', '?')
    closed = (d_at_rfire is not None and min_d_after < CLOSE_FRAC * d_at_rfire)
    med_asp = float(np.median(aspects_after)) if aspects_after else 999
    aggressive = closed and med_asp < ASPECT_MAX
    if oc == 'win' and bsh > 0 and rsh > 0 and aggressive:
        for _ in range(12):  # post-kill coast
            env.tacview_logger.log_flight_data()
            a, _ = model.predict(obs, deterministic=True)
            try: obs, _, _, _, info = env.step(a)
            except Exception: break
        env.tacview_logger.save_logs()
        found += 1
        name = (f"aggr_win{found}_seed{seed}_B{bsh}R{rsh}"
                f"_rfire{d_at_rfire/1e3:.0f}km_closeto{min_d_after/1e3:.0f}km"
                f"_asp{med_asp:.0f}.acmi")
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, name))
        print(f"  >>> KEPT {name}")
    else:
        tag = oc if oc != 'win' else f"win B{bsh}/R{rsh} closed={closed} asp={med_asp:.0f}"
        print(f"  seed {seed}: {tag} -- skip")
    seed += 1
print(f"\n{found} aggressive blue-win replays saved to two_sided_duels/")
