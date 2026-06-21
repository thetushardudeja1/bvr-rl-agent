import sys, os
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.config import baseEnv_conf
from experiments.v2.selfplay_env_v2 import SelfPlayBVR

def run_test():
    baseEnv_conf.opponent_path = 'experiments/v2/seed_clean'
    env = SelfPlayBVR(baseEnv_conf)
    
    print("Loading 47% seed model...")
    model = PPO.load('experiments/v2/seed_clean', device='cpu')
    
    print("Resetting environment...")
    obs, _ = env.reset()
    
    print("\n--- Running 20 steps of simulation ---")
    print(f"{'Step':>4} | {'Raw Action (Pitch)':>18} | {'Smoothed (Pitch)':>18} | {'Penalty':>10} | {'Base Reward':>12}")
    print("-" * 75)
    
    for i in range(20):
        # The agent predicts an action
        action, _ = model.predict(obs, deterministic=True)
        
        # We manually calculate what the environment will do for logging
        if env.prev_action is None:
            prev_act = action.copy()
        else:
            prev_act = env.prev_action.copy()
            
        smoothed_action = 0.8 * prev_act + 0.2 * action
        action_penalty = 0.001 * np.sum(np.abs(smoothed_action - prev_act))
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Base reward is the total reward plus the penalty we subtracted
        base_reward = reward + action_penalty
        
        print(f"{i:>4} | {action[0]:>18.4f} | {smoothed_action[0]:>18.4f} | {action_penalty:>10.6f} | {base_reward:>12.6f}")
        
        if done:
            print("Episode finished early.")
            break

if __name__ == "__main__":
    run_test()
