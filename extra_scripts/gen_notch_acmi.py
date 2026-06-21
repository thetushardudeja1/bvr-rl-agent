"""
gen_notch_acmi.py — Tacview replays of the notch-trained agent, labeled by the
evasion blue actually used, so the varied tactics are visible:
   BEAM : blue turned ~perpendicular and the red missile dropped lock (notch)
   DRAG : blue extended and the red missile fell short (kinematic/energy)
Saves one clear BEAM-only, one VARIED (both), and any BLUE-WIN found.
"""
import sys, os, shutil
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.loggers import TacviewLogger

MODEL = 'experiments/rir/rir_notch'
WOUT  = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/notch_demo'
os.makedirs(WOUT, exist_ok=True)
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

want = {'beam': 1, 'varied': 1, 'win': 1}
got  = {'beam': 0, 'varied': 0, 'win': 0}
print("scanning for labeled evasion replays...")
seed = 6000
while sum(got.values()) < sum(want.values()) and seed < 6120:
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)
    seen = {}; n_notch = n_kin = 0
    done = trunc = False
    while not (done or trunc):
        env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        for m in env.red_agent.ammo.values():
            if m.target_lost and id(m) not in seen:
                if m.notch_count > m.missile_performance.notch_count:
                    seen[id(m)] = 'n'; n_notch += 1
                else:
                    seen[id(m)] = 'k'; n_kin += 1
    oc = info.get('outcome', '?')
    for _ in range(12):                       # post-event coast
        env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        try: obs, _, _, _, info = env.step(a)
        except Exception: break

    label = None
    if oc == 'win' and got['win'] < want['win']:
        label = 'win'
    elif n_notch > 0 and n_kin > 0 and got['varied'] < want['varied']:
        label = 'varied'
    elif n_notch > 0 and n_kin == 0 and got['beam'] < want['beam']:
        label = 'beam'
    if label:
        env.tacview_logger.save_logs()
        got[label] += 1
        name = f"notch_{label}_seed{seed}_beam{n_notch}_drag{n_kin}_{oc}.acmi"
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, name))
        print(f"  >>> SAVED {name}")
    else:
        print(f"  seed {seed}: {oc} beam={n_notch} drag={n_kin} -- skip")
    seed += 1
print(f"\nsaved to notch_demo/: {got}")
