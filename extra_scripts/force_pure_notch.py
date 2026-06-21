import sys, os, shutil
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.loggers import TacviewLogger

MODEL = 'experiments/rir/rir_notch2_best'

# Force a short-range engagement where dragging is impossible
baseEnv_conf.initial_conditions = {
    'blue': {'lat': 0.0, 'lon': 0.0, 'alt': 8000, 'heading': 0, 'speed': 250},
    'red':  {'lat': 0.2, 'lon': 0.0, 'alt': 8000, 'heading': 180, 'speed': 250} # ~22km apart
}
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

seed = 9999
env = BVRBase(baseEnv_conf)
obs, _ = env.reset(seed=seed)

# Override positions just in case
env.blue_agent.simObj.reset(lat=0.0, long=0.0, alt=8000, vel=250, heading=0)
env.red_agent.simObj.reset(lat=0.15, long=0.0, alt=8000, vel=250, heading=180)

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
WOUT = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/demo_replays'
os.makedirs(WOUT, exist_ok=True)
name = f"close_range_pure_notch.acmi"
shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
            os.path.join(WOUT, name))
print(f"Saved to {name}")
