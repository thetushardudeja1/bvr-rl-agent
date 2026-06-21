import sys, os, shutil
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.loggers import TacviewLogger

MODEL = 'experiments/rir/rir_notch2_best'
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

print("Searching for multiple PURE NOTCH wins... (no dragging)")
found_count = 0
for seed in range(8000, 15000):
    if found_count >= 5:
        break
        
    env = BVRBase(baseEnv_conf)
    obs, _ = env.reset(seed=seed)
    
    seen = {}
    match_notches = 0
    match_kinematics = 0
    
    done = trunc = False
    while not (done or trunc):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        
        for m in env.red_agent.ammo.values():
            if m.target_lost and id(m) not in seen:
                if m.notch_count > m.missile_performance.notch_count:
                    seen[id(m)] = 'n'
                    match_notches += 1
                else:
                    seen[id(m)] = 'k'
                    match_kinematics += 1

    oc = info.get('outcome', '?')
    
    if oc == 'win' and match_notches > 0 and match_kinematics == 0:
        print(f"\n>>> FOUND PURE NOTCH WIN AT SEED {seed} <<<")
        print(f"Notches: {match_notches}, Kinematics (Drag): {match_kinematics}")
        
        # Now we regenerate it to save the Tacview
        print("Regenerating with Logger...")
        env = BVRBase(baseEnv_conf)
        obs, _ = env.reset(seed=seed)
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
        WOUT = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/demo_replays/pure_notch_wins'
        os.makedirs(WOUT, exist_ok=True)
        name = f"pure_notch_win_seed{seed}.acmi"
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, name))
        print(f"Saved to {name}")
        found_count += 1

print(f"Finished finding {found_count} pure notch wins.")
