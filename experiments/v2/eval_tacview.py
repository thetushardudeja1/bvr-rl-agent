import os, sys, glob, shutil
sys.path.insert(0, os.path.expanduser("~/bvrgym_project_new"))

from stable_baselines3 import PPO
from jsb_gym.envs.config import baseEnv_conf
from experiments.v2.selfplay_env_v2 import SelfPlayBVR

class TacviewSelfPlayBVR(SelfPlayBVR):
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if getattr(self.conf, 'tacview_output', False):
            from jsb_gym.utils.loggers import TacviewLogger
            self.tacview_logger = TacviewLogger(self)
        return obs, info

    def step(self, action):
        ret = super().step(action)
        if hasattr(self, 'tacview_logger') and self.tacview_logger is not None:
            self.tacview_logger.log_flight_data()
        return ret

def main():
    print("Loading selfplay_best model...")
    model_path = os.path.expanduser("~/bvrgym_project_new/experiments/v2/models/selfplay_best.zip")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model = PPO.load(model_path, device="cpu")
    
    # Configure tacview
    baseEnv_conf.tacview_output = True
    baseEnv_conf.tacview_output_dir = os.path.expanduser("~/bvrgym_project_new/experiments/v2/tacview")
    baseEnv_conf.opponent_path = 'BT'  
    os.makedirs(baseEnv_conf.tacview_output_dir, exist_ok=True)
    
    print("Initializing environment...")
    env = TacviewSelfPlayBVR(conf=baseEnv_conf)
    
    obs, _ = env.reset()
    done = False
    step = 0
    
    print("Running episode...")
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, rew, done, trunc, info = env.step(action)
        step += 1
        if done or trunc:
            break
            
    print(f"Episode finished in {step} steps. Outcome: {info.get('outcome', 'unknown')}")
    
    if hasattr(env, 'tacview_logger') and env.tacview_logger:
        env.tacview_logger.save_logs()
        
    env.close()
    
    tacview_files = glob.glob(os.path.join(baseEnv_conf.tacview_output_dir, "*.acmi"))
    if tacview_files:
        latest_file = max(tacview_files, key=os.path.getctime)
        downloads_dir = "/mnt/c/Users/TUSHAR/Downloads"
        dest_path = os.path.join(downloads_dir, "best_model_replay.acmi")
        shutil.copy2(latest_file, dest_path)
        print(f"Copied Tacview file to: {dest_path}")
    else:
        print("No tacview file found!")

if __name__ == '__main__':
    main()
