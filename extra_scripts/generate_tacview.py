"""
generate_tacview.py
===================
Smart TacView generator — stops IMMEDIATELY on first WIN.
Saves ONLY the winning episode's TacView files. No rerun.

Usage:
    export PYTHONPATH=.
    python generate_tacview.py
    python generate_tacview.py trained/BVRBase_best.zip
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents


def clear_tacview():
    tacview_dir = baseEnv_conf.tacview_output_dir
    if os.path.exists(tacview_dir):
        shutil.rmtree(tacview_dir)
    os.makedirs(tacview_dir, exist_ok=True)


def run_episode(model):
    env = BVRBase(baseEnv_conf)
    obs, _ = env.reset()

    done      = False
    truncated = False
    step      = 0
    reward    = 0
    min_dist  = float('inf')

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        env.log_tacview()
        step += 1

        dist = dinstance_between_agents(env.blue_agent, env.red_agent)
        if dist < min_dist:
            min_dist = dist

    env.log_tacview()

    if reward > 0:
        result = 'WIN'
    elif truncated:
        result = 'TIMEOUT'
    else:
        result = 'LOSS'

    return {'result': result, 'reward': reward, 'steps': step, 'min_dist': min_dist / 1000}


def find_model(model_path=None):
    if model_path is not None:
        return model_path.replace('.zip', '')
    for candidate in ['trained/BVRBase_best', 'trained/BVRBase_15M_final', 'trained/BVRBase_5M_fixed']:
        if os.path.exists(candidate + '.zip'):
            return candidate
    print("\nERROR: No trained model found!")
    sys.exit(1)


def generate_best_tacview(model_path=None, max_attempts=50):
    model_path = find_model(model_path)

    print("\n" + "=" * 65)
    print("  BVR GYM — Smart TacView Generator")
    print("  Stops IMMEDIATELY on WIN — guaranteed winning episode saved")
    print("=" * 65)
    print(f"  Model        : {model_path}.zip")
    print(f"  Max attempts : {max_attempts}")
    print("=" * 65)

    print("\n  Loading model...")
    model = PPO.load(model_path)
    print("  Model loaded!\n")

    print(f"  {'Attempt':>8}  {'Result':>8}  {'Steps':>7}  {'MinDist':>9}  {'Reward':>8}")
    print("  " + "-" * 50)

    for attempt in range(1, max_attempts + 1):

        # Clear BEFORE episode — so winning episode files are never cleared after
        clear_tacview()

        ep = run_episode(model)

        print(f"  {attempt:>8}  {ep['result']:>8}  {ep['steps']:>7}  "
              f"{ep['min_dist']:>8.1f}km  {ep['reward']:>8.1f}")

        if ep['result'] == 'WIN':
            # Files from this episode are saved — stop immediately
            print(f"\n  ✅ WIN on attempt {attempt} — TacView saved!")
            print(f"     Steps to kill : {ep['steps']}")
            print(f"     Closest dist  : {ep['min_dist']:.1f} km")
            print(f"     Reward        : {ep['reward']:.1f}")
            print("\n  TacView files:")
            tacview_dir = baseEnv_conf.tacview_output_dir
            for f in sorted(os.listdir(tacview_dir)):
                print(f"    {tacview_dir}/{f}")
            print("\n  Copy to Windows:")
            print("  mkdir -p /mnt/c/Users/TUSHAR/Downloads/tacview_win")
            print("  cp ~/bvrgym_project/data_output/tacview/* /mnt/c/Users/TUSHAR/Downloads/tacview_win/")
            print("\n  HOW TO VIEW:")
            print("  1. Open TacView → File → Open → select ALL CSV files")
            print("  2. Press F7 for 2D, F8 for 3D, F to fit all objects")
            print("  3. Press Play ▶")
            print("=" * 65 + "\n")
            return  # EXIT immediately — winning files preserved

    print(f"\n  No wins in {max_attempts} attempts. Try again.")


if __name__ == "__main__":
    args = sys.argv[1:]
    model_path   = None
    max_attempts = 50
    for arg in args:
        if arg.startswith('--max_attempts='):
            max_attempts = int(arg.split('=')[1])
        elif not arg.startswith('--'):
            model_path = arg
    generate_best_tacview(model_path, max_attempts)
