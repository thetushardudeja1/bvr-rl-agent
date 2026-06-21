"""
generate_tacview.py
===================
Generates a Tacview (.acmi) file demonstrating the ORIGINAL autopilot
performing a sequence of maneuvers, followed by RL-like jitter.
"""

import os, sys, math, time
import numpy as np

ROOT = os.path.expanduser("~/bvrgym_project_new")
sys.path.insert(0, ROOT)

from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf

def hdg_to_action(hdg_deg):
    return hdg_deg / 180.0 - 1.0

def alt_to_action(alt_m):
    return (alt_m - 3000.0) / 4500.0 - 1.0

def get_initial_state(env):
    ac = env.blue_agent.simObj
    return ac.get_psi(), ac.get_altitude()

def main():
    print("Generating Tacview file for ORIGINAL autopilot maneuvers...")
    
    # Enable tacview
    baseEnv_conf.tacview_output = True
    baseEnv_conf.tacview_output_dir = os.path.join(ROOT, "experiments", "v2", "tacview")
    os.makedirs(baseEnv_conf.tacview_output_dir, exist_ok=True)
    
    # The BVRGym tacview logger uses a hardcoded 'logs' directory by default sometimes, 
    # but setting tacview_output to True is the main switch.
    env = BVRBase(baseEnv_conf)
    obs, _ = env.reset()
    
    init_hdg, init_alt = get_initial_state(env)
    
    print(f"Initial State: Heading={init_hdg:.1f}°, Altitude={init_alt:.1f}m")
    
    total_steps = 1000
    
    rng_action = np.array([0.0, 0.0, 0.5])
    
    for step in range(total_steps):
        if step < 200:
            if step == 0: print("Phase 1: Straight and Level")
            action = np.array([hdg_to_action(init_hdg), alt_to_action(init_alt), 0.5])
            
        elif step < 400:
            if step == 200: print("Phase 2: 90° Right Turn")
            target = (init_hdg + 90) % 360
            action = np.array([hdg_to_action(target), alt_to_action(init_alt), 0.5])
            
        elif step < 600:
            if step == 400: print("Phase 3: Climb (+1500m)")
            target = (init_hdg + 90) % 360
            action = np.array([hdg_to_action(target), alt_to_action(init_alt + 1500), 0.8])
            
        elif step < 800:
            if step == 600: print("Phase 4: 180° Left Turn & Descend (-1500m)")
            target = (init_hdg - 90) % 360
            action = np.array([hdg_to_action(target), alt_to_action(init_alt - 1500), 0.1])
            
        else:
            if step == 800: print("Phase 5: RL-style Rapid Jitter (The 'Wobble')")
            # Change random target every 2 steps to simulate bad RL policy
            if step % 2 == 0:
                rng_action = np.random.uniform(-1, 1, size=3)
                rng_action[2] = abs(rng_action[2]) # keep thrust positive
            action = rng_action
            
        obs, rew, done, trunc, info = env.step(action)
        if done or trunc:
            print(f"Episode ended early at step {step}")
            if env.tacview_logger:
                env.tacview_logger.save_logs()
            break
            
    if env.tacview_logger and not env.done:
        env.tacview_logger.save_logs()
        
    env.close()
    print("Run completed! Check your logs folder for the .acmi Tacview file.")

if __name__ == '__main__':
    main()
