"""
collect_bt_data.py
==================
Generate the imitation (expert) dataset by running BVREnv_BTvsBT — where the
BLUE agent is the behavior-tree expert — and recording (observation, action)
pairs from blue's perspective.

  observation : env.state  (stacked frames, shape = observation_shape)
  action      : blue BT's [heading, altitude, throttle] in NN-normalized [-1,1]

Usage:
    python -m experiments.rir.collect_bt_data --episodes 30 --out experiments/rir/bt_dataset.npz
"""
import sys, argparse, time
import numpy as np
sys.path.insert(0, '.')

from jsb_gym.envs.BaseEnv import BVREnv_BTvsBT
from jsb_gym.envs.config import baseEnv_conf
from experiments.rir.action_utils import bt_action_to_nn


def collect(n_episodes, seed0=0):
    obs_list, act_list = [], []
    ep_lens = []
    for ep in range(n_episodes):
        env = BVREnv_BTvsBT(baseEnv_conf)
        state, _ = env.reset(seed=seed0 + ep)
        limits = env.blue_agent.simObj.conf.aircraft_limits
        done = trunc = False
        steps = 0
        while not (done or trunc):
            obs_before = np.array(state, dtype=np.float32)   # (40,15)
            state, _, done, trunc, _ = env.step(None)
            # blue BT just ticked inside step -> read the action it applied
            a = bt_action_to_nn(env.blue_agent.BT.heading,
                                env.blue_agent.BT.altitude, limits)
            obs_list.append(obs_before)
            act_list.append(np.array(a, dtype=np.float32))
            steps += 1
        ep_lens.append(steps)
        print(f"  ep {ep+1}/{n_episodes}: {steps} steps  (total pairs={len(obs_list)})")
    return np.array(obs_list), np.array(act_list), ep_lens


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--out', type=str, default='experiments/rir/bt_dataset.npz')
    args = p.parse_args()

    t0 = time.time()
    obs, acts, ep_lens = collect(args.episodes)
    np.savez_compressed(args.out, obs=obs, acts=acts)
    dt = time.time() - t0
    print("\n" + "=" * 55)
    print(f"  Collected {len(obs):,} (obs, action) pairs in {dt:.0f}s")
    print(f"  obs shape : {obs.shape}   acts shape : {acts.shape}")
    print(f"  action ranges  heading[{acts[:,0].min():.2f},{acts[:,0].max():.2f}]"
          f"  alt[{acts[:,1].min():.2f},{acts[:,1].max():.2f}]"
          f"  thr[{acts[:,2].min():.2f},{acts[:,2].max():.2f}]")
    print(f"  saved -> {args.out}")
    print("=" * 55)


if __name__ == "__main__":
    main()
