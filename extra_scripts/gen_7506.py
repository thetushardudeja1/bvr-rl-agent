import sys, os, shutil
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.loggers import TacviewLogger

MODEL = 'experiments/rir/rir_notch2_best'
WOUT  = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/demo_replays'
os.makedirs(WOUT, exist_ok=True)
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

seed = 7506
env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
env.tacview_logger = TacviewLogger(env)
done = trunc = False

while not (done or trunc):
    env.tacview_logger.log_flight_data()
    a, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, info = env.step(a)

for _ in range(12):                       # post-kill coast
    env.tacview_logger.log_flight_data()
    a, _ = model.predict(obs, deterministic=True)
    try: obs, _, _, _, info = env.step(a)
    except Exception: break

env.tacview_logger.save_logs()
name = f"blue_notches_and_wins_seed{seed}.acmi"
shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
            os.path.join(WOUT, name))
print(f"  >>> SAVED WINNING NOTCH DEMO: {name}")
