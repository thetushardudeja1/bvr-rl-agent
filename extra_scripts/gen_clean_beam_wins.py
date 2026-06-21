"""
gen_clean_beam_wins.py — the specific case the supervisor wanted: blue does NOT
turn back. Keep only episodes where:
  (1) blue WINS,
  (2) blue defeats red's missile by BEAMING (notch) with NO drag escapes,
  (3) blue keeps its nose FORWARD — while a red missile is inbound, red stays in
      blue's forward hemisphere (median |relative bearing to red| < FWD_MAX),
      i.e. blue beams ~90 deg and presses, instead of reversing 180 deg to run.
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
WOUT    = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/notch_clean_beam_wins'
FWD_MAX = 80.0        # blue keeps red within +-80 deg of the nose while threatened
WANT    = 5
os.makedirs(WOUT, exist_ok=True)
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

found, seed = 0, 7000
print("scanning for clean-beam (no turn-back) blue wins...")
while found < WANT and seed < 7600:
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)
    bsh = rsh = 0; pb = pr = False; min_d = 1e9
    seen = {}; n_notch = n_kin = 0
    fwd_bearings = []                     # |rel bearing to red| while red missile up
    done = trunc = False
    while not (done or trunc):
        env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        b, r = env.blue_agent, env.red_agent
        min_d = min(min_d, dinstance_between_agents(b, r))
        ba, ra = b.is_own_missile_active(), r.is_own_missile_active()
        bsh += (ba and not pb); rsh += (ra and not pr); pb, pr = ba, ra
        if ra:                                          # threat inbound
            fwd_bearings.append(abs(relative_bearing_between_agents(b, r)))
        for m in r.ammo.values():
            if m.target_lost and id(m) not in seen:
                if m.notch_count > m.missile_performance.notch_count:
                    seen[id(m)] = 'n'; n_notch += 1
                else:
                    seen[id(m)] = 'k'; n_kin += 1
    oc = info.get('outcome', '?')
    med_fwd = float(np.median(fwd_bearings)) if fwd_bearings else 999
    clean = (oc == 'win' and rsh > 0 and n_notch >= 1 and n_kin == 0 and med_fwd < FWD_MAX)
    if clean:
        for _ in range(12):
            env.tacview_logger.log_flight_data()
            a, _ = model.predict(obs, deterministic=True)
            try: obs, _, _, _, info = env.step(a)
            except Exception: break
        env.tacview_logger.save_logs()
        found += 1
        name = (f"clean_beam_win{found}_seed{seed}_B{bsh}R{rsh}"
                f"_beam{n_notch}_fwd{med_fwd:.0f}_min{min_d/1e3:.0f}km.acmi")
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, name))
        print(f"  >>> SAVED {name}")
    else:
        print(f"  seed {seed}: {oc} R{rsh} beam{n_notch} drag{n_kin} fwd{med_fwd:.0f} -- skip")
    seed += 1
print(f"\n{found} clean-beam (no turn-back) blue wins saved to notch_clean_beam_wins/")
