"""
collect_bt_parallel.py
======================
Improved demo collection for the RIR warmup:
  - parallel workers (multiprocessing Pool)
  - randomized engagement geometry per episode (env IC randomization)
  - kill-terminated episodes (no post-kill garbage in the dataset)
Replaces the old 30-episode sequential, single-frozen-scenario collection.

Usage:
    python -m experiments.rir.collect_bt_parallel --episodes 400 --workers 12 \
        --out experiments/rir/bt_dataset.npz
"""
import sys, argparse, time, os
import numpy as np
import multiprocessing as mp
sys.path.insert(0, '.')


def collect_worker(args):
    worker_id, episodes, seed0, wins_only = args
    from jsb_gym.envs.BaseEnv import BVREnv_BTvsBT
    from jsb_gym.envs.config import baseEnv_conf
    from experiments.rir.action_utils import bt_action_to_nn

    obs_list, act_list, outcomes = [], [], []
    env = BVREnv_BTvsBT(baseEnv_conf)
    for ep in range(episodes):
        state, _ = env.reset(seed=seed0 + ep)
        limits = env.blue_agent.simObj.conf.aircraft_limits
        done = trunc = False
        ep_obs, ep_act = [], []          # buffer this episode, commit on outcome
        while not (done or trunc):
            obs_before = np.array(state, dtype=np.float32)
            state, _, done, trunc, info = env.step(None)
            a = bt_action_to_nn(env.blue_agent.BT.heading,
                                env.blue_agent.BT.altitude, limits,
                                bt_launch=env.blue_agent.BT.launch_missile)
            ep_obs.append(obs_before)
            ep_act.append(np.array(a, dtype=np.float32))
        oc = info.get('outcome', '?')
        outcomes.append(oc)
        # win-biased imitation: clone only the BT's WINNING (aggressive,
        # kill-producing) trajectories so BC instils a kill prior, not the
        # passive timeout behaviour that dominated the unfiltered set.
        if (not wins_only) or oc == 'win':
            obs_list.extend(ep_obs)
            act_list.extend(ep_act)
    return np.array(obs_list), np.array(act_list), outcomes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=400)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--out', type=str, default='experiments/rir/bt_dataset.npz')
    p.add_argument('--wins_only', action='store_true',
                   help='keep only BT winning episodes (kill prior for BC)')
    args = p.parse_args()

    per_worker = args.episodes // args.workers
    jobs = [(w, per_worker, 10_000 * w, args.wins_only) for w in range(args.workers)]

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(collect_worker, jobs)

    obs = np.concatenate([r[0] for r in results])
    acts = np.concatenate([r[1] for r in results])
    outcomes = [o for r in results for o in r[2]]

    np.savez_compressed(args.out, obs=obs, acts=acts)
    n_ep = len(outcomes)
    from collections import Counter
    c = Counter(outcomes)
    print("\n" + "=" * 60)
    print(f"  {n_ep} episodes, {len(obs):,} (obs,action) pairs in {time.time()-t0:.0f}s")
    print(f"  outcomes: {dict(c)}")
    print(f"  action ranges  heading[{acts[:,0].min():.2f},{acts[:,0].max():.2f}]"
          f"  alt[{acts[:,1].min():.2f},{acts[:,1].max():.2f}]")
    print(f"  saved -> {args.out}")
    print("=" * 60)


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
