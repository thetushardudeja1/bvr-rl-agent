"""
train_rir.py
============
RIR / DAPG training orchestrator:
  Phase 1 — value-function warmup (actor frozen) so PPO starts with sane
            advantages (prevents value-driven destruction of the warm-start).
  Phase 2 — PPO with DAPG auxiliary BC loss on the BT demonstrations
            (decaying bc_coef) — the proven fix for BC->RL policy collapse.

Assumes BC already run (experiments.rir.behavior_clone) producing bc_model.zip
and the demonstration dataset bt_dataset.npz.

Usage:
    python -m experiments.rir.train_rir \
        --bc experiments/rir/bc_model --data experiments/rir/bt_dataset.npz \
        --steps 50000000 --out experiments/rir/rir_model
"""
import sys, argparse, os, time
import numpy as np
sys.path.insert(0, '.')

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.envs.vec_env import ChunkSubprocVecEnv
from experiments.rir.ppo_aeg import PPO_AEG
from experiments.rir.callbacks import RIRLogger


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bc', type=str, default='experiments/rir/bc_model')
    p.add_argument('--warmstart', type=str, default=None,
                   help='continue from a prior RIR checkpoint instead of the BC model')
    p.add_argument('--data', type=str, default='experiments/rir/bt_dataset.npz')
    p.add_argument('--steps', type=int, default=15_000_000)
    p.add_argument('--n_envs', type=int, default=360)
    p.add_argument('--n_steps', type=int, default=128)
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--bc_start', type=float, default=1.0)   # DAPG aux BC weight start
    p.add_argument('--bc_end', type=float, default=0.1)     # decays to this (keeps anchor)
    p.add_argument('--ent_coef', type=float, default=0.003)
    p.add_argument('--gamma', type=float, default=0.999)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--value_warmup', type=int, default=1_000_000,
                   help='steps to fit value head with actor FROZEN before RL')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--out', type=str, default='experiments/rir/rir_model')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--red_doctrine', action='store_true',
                   help='randomize red firing range/energy per episode (threat diversity)')
    args = p.parse_args()

    # enable doctrine randomization in the shared env config BEFORE forking the
    # vec-env workers (fork inherits this module state into each subprocess)
    if args.red_doctrine:
        baseEnv_conf.red_doctrine_randomize = True
        print(f"  red doctrine randomization ON  scale~U{baseEnv_conf.red_launch_scale_range}")

    run_name = args.run_name or f"RIR_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir  = os.path.join('runs', run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('experiments/rir/checkpoints', exist_ok=True)
    print(f"  logging -> {log_dir}  (tensorboard + csv + stdout)")

    vec_env = make_vec_env(BVRBase, n_envs=args.n_envs,
                           vec_env_cls=ChunkSubprocVecEnv,
                           env_kwargs={'conf': baseEnv_conf})

    model = PPO_AEG("MlpPolicy", vec_env, device=args.device, verbose=0,
                    learning_rate=args.lr,
                    n_steps=args.n_steps, batch_size=args.batch,
                    ent_coef=args.ent_coef, gamma=args.gamma,
                    bc_coef_start=args.bc_start, bc_coef_end=args.bc_end)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # warm-start: load weights into the live policy. Prefer a prior RIR
    # checkpoint (--warmstart) to continue progress; else the BC model.
    if args.warmstart:
        src = PPO_AEG.load(args.warmstart, device=args.device,
                           custom_objects={'lr_schedule': lambda _: args.lr,
                                           'clip_range': lambda _: 0.2})
        print(f"  continuing from RIR checkpoint: {args.warmstart}")
    else:
        src = PPO.load(args.bc, device=args.device)
        print(f"  warm-starting from BC model: {args.bc}")
    model.policy.load_state_dict(src.policy.state_dict())

    # DAPG demonstrations (the BT dataset used for BC)
    d = np.load(args.data)
    model.set_demonstrations(d['obs'], d['acts'])

    print(f"  warm-start loaded; lr={args.lr}, bc_coef={args.bc_start}->{args.bc_end}, "
          f"ent={args.ent_coef}, gamma={args.gamma}")

    # save_freq counts per-env collection steps: a checkpoint lands every
    # save_freq * n_envs timesteps. 500k//n_envs => one ~120MB zip per 500k
    # steps (~6 min) instead of every ~1.3 rollouts (30s, ~40GB per run).
    # Best-model saving is separate (RIRLogger ratchet) and unaffected.
    ckpt = CheckpointCallback(save_freq=max(1, 500_000 // args.n_envs),
                              save_path='experiments/rir/checkpoints/',
                              name_prefix='rir')
    rir_logger = RIRLogger(save_path=args.out, window=100, print_freq=20)

    # actor params (frozen during value warmup); value path stays trainable
    actor_params = (list(model.policy.mlp_extractor.policy_net.parameters())
                    + list(model.policy.action_net.parameters())
                    + [model.policy.log_std])

    # ================= PHASE 1: value-function warmup =================
    if args.value_warmup > 0:
        print(f"  PHASE 1: value warmup ({args.value_warmup:,} steps, actor FROZEN, BC off)")
        for prm in actor_params:
            prm.requires_grad_(False)
        model._bc_active = False
        model.learn(total_timesteps=args.value_warmup,
                    callback=[rir_logger], progress_bar=False,
                    reset_num_timesteps=True)
        for prm in actor_params:
            prm.requires_grad_(True)
        print("  value head warmed up; unfreezing actor.")

    # ================= PHASE 2: RL + DAPG BC loss =====================
    print(f"  PHASE 2: PPO + DAPG BC loss ({args.steps:,} steps)")
    model._bc_active = True
    model._bc_t0 = model.num_timesteps
    model._bc_total = args.steps
    model.learn(total_timesteps=args.steps,
                callback=[ckpt, rir_logger], progress_bar=False,
                reset_num_timesteps=False)

    model.save(args.out)
    print(f"  ✅ RIR model saved -> {args.out}.zip   (best -> {args.out}_best.zip)")
    print(f"     logs in {log_dir}")
    vec_env.close()


if __name__ == "__main__":
    import multiprocessing
    # 'fork' is Linux-only; Windows/macOS-default must use 'spawn'. Pick whatever
    # the platform actually supports so the same file runs everywhere.
    methods = multiprocessing.get_all_start_methods()
    multiprocessing.set_start_method("fork" if "fork" in methods else "spawn",
                                     force=True)
    main()
