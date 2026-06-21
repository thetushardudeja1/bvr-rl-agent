"""
gen_maneuver_wins.py — wins where blue ACTIVELY MANEUVERS to defeat red's
missiles (beam/notch), NOT energy-bleed, NOT turning away and running:
  (1) blue WINS,
  (2) red actually shoots (rsh >= 2),
  (3) EVERY red missile blue defeated was beaten by the NOTCH maneuver
      (n_notch >= 1 and n_kin == 0 — no kinematic/energy/drag escapes),
  (4) blue keeps red in its forward hemisphere (median |rel bearing| < FWD_MAX)
      so it is beaming ~90 deg and pressing, not reversing 180 deg.
Post-kill coast so the kill is visible.
"""
import sys, os, shutil, numpy as np
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents, relative_bearing_between_agents
from jsb_gym.utils.loggers import TacviewLogger

MODEL   = 'experiments/rir/rir_notch2_best'
WOUT    = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/self play win'
FWD_MAX = 70.0        # stricter: blue keeps red well forward, never reverses
WANT    = 8
os.makedirs(WOUT, exist_ok=True)
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

found, seed = 0, 7550
print("scanning for maneuver-evasion (beam-only, no turn-back) blue wins...")
while found < WANT and seed < 8600:
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)
    rsh = 0; pr = False
    seen = {}; n_notch = n_kin = 0; min_call = 1e9
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
                min_call = min(min_call, m.target_norm)
            if m.target_lost and id(m) not in seen:
                seen[id(m)] = ('n' if m.notch_count > m.missile_performance.notch_count else 'k')
                if seen[id(m)] == 'n': n_notch += 1
                else: n_kin += 1
    oc = info.get('outcome', '?')
    med_fwd = float(np.median(fwd_bearings)) if fwd_bearings else 999
    ok = (oc == 'win' and rsh >= 2 and n_notch >= 1 and n_kin == 0 and med_fwd < FWD_MAX)
    if ok:
        for _ in range(12):
            env.tacview_logger.log_flight_data()
            a, _ = model.predict(obs, deterministic=True)
            try: obs, _, _, _, info = env.step(a)
            except Exception: break
        env.tacview_logger.save_logs()
        found += 1
        name = (f"maneuver_win{found}_seed{seed}_R{rsh}_beam{n_notch}"
                f"_nearmiss{min_call:.0f}m_fwd{med_fwd:.0f}.acmi")
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, name))
        print(f"  >>> SAVED {name}")
    else:
        print(f"  seed {seed}: {oc} R{rsh} beam{n_notch} drag{n_kin} fwd{med_fwd:.0f} -- skip")
    seed += 1
print(f"\n{found} maneuver-evasion blue wins saved to notch_maneuver_wins/")
