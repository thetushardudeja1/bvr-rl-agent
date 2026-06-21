"""
gen_close_call_wins.py — the dramatic case: a TOUGH, CLOSE-CALL blue win.
Keep only episodes where:
  (1) blue WINS,
  (2) red puts up a real fight: fires >= RSH_MIN missiles,
  (3) at least one red missile comes within CALL_MAX metres of blue (a near miss,
      just outside the 300 m lethal radius) — the close call,
  (4) blue defeats it by BEAMING and keeps its nose FORWARD (no turn-back):
      n_notch>=1 and median |rel bearing to red while threatened| < FWD_MAX.
Saved sorted-by-drama in the filename (closest call first). Post-kill coast.
"""
import sys, os, shutil, numpy as np
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents, relative_bearing_between_agents
from jsb_gym.utils.loggers import TacviewLogger

MODEL    = 'experiments/rir/rir_notch2_best'
WOUT     = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/notch_close_call_wins'
RSH_MIN  = 3            # red fires at least this many (tough fight)
CALL_MAX = 1500.0       # a red missile passed within this many metres of blue
FWD_MAX  = 90.0         # blue keeps red forward (no 180 turn-back)
WANT     = 5
os.makedirs(WOUT, exist_ok=True)
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

found, seed = 0, 7000
print("scanning for tough, close-call, no-turn-back blue wins...")
while found < WANT and seed < 7800:
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)
    rsh = 0; pr = False
    seen = {}; n_notch = n_kin = 0
    min_call = 1e9                          # closest any red missile got to blue
    fwd_bearings = []
    done = trunc = False
    while not (done or trunc):
        env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        b, r = env.blue_agent, env.red_agent
        ra = r.is_own_missile_active()
        rsh += (ra and not pr); pr = ra
        if ra:
            fwd_bearings.append(abs(relative_bearing_between_agents(b, r)))
        for m in r.ammo.values():
            if m.is_active() and m.target_norm is not None:
                min_call = min(min_call, m.target_norm)   # missile-to-blue range
            if m.target_lost and id(m) not in seen:
                seen[id(m)] = ('n' if m.notch_count > m.missile_performance.notch_count else 'k')
                if seen[id(m)] == 'n': n_notch += 1
                else: n_kin += 1
    oc = info.get('outcome', '?')
    med_fwd = float(np.median(fwd_bearings)) if fwd_bearings else 999
    ok = (oc == 'win' and rsh >= RSH_MIN and n_notch >= 1
          and min_call < CALL_MAX and med_fwd < FWD_MAX)
    if ok:
        for _ in range(12):
            env.tacview_logger.log_flight_data()
            a, _ = model.predict(obs, deterministic=True)
            try: obs, _, _, _, info = env.step(a)
            except Exception: break
        env.tacview_logger.save_logs()
        found += 1
        name = (f"closecall_win{found}_seed{seed}_R{rsh}"
                f"_nearmiss{min_call:.0f}m_beam{n_notch}_drag{n_kin}_fwd{med_fwd:.0f}.acmi")
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, name))
        print(f"  >>> SAVED {name}")
    else:
        print(f"  seed {seed}: {oc} R{rsh} nearmiss{min_call:.0f}m beam{n_notch} fwd{med_fwd:.0f} -- skip")
    seed += 1
print(f"\n{found} tough close-call blue wins saved to notch_close_call_wins/")
