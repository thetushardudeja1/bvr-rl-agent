"""
================================================================================
 RUN_train.py  --  PRESS THE ▶ "RUN" BUTTON IN VS CODE TO TRAIN THE BLUE AGENT
================================================================================
No terminal typing needed. Open this file in VS Code and click Run (▶) or F5.

It trains the Blue PPO agent on the notch-capable environment, warm-started from
the behaviour-cloning model and anchored to the BT demonstrations (the "RIR"
pipeline). Checkpoints + the best model are written under ./experiments/rir/.

IMPORTANT — this is heavy compute. Edit the CONFIG block to match your machine:
  * N_ENVS : parallel simulators. More = faster but more RAM/CPU.
            Use ~ (number of CPU cores). Keep <= ~8-16 on a laptop.
  * STEPS  : total training timesteps. 2,000,000 ~ a quick run; the paper model
            used 12,000,000 (hours). Start small to confirm it runs.
Progress prints in the VS Code Run panel; a TensorBoard log is also written.
================================================================================
"""
import os, sys, subprocess, multiprocessing

# Windows consoles default to cp1252 and crash on the emoji/arrows the sim
# prints. Force UTF-8 everywhere (this process + the training subprocess).
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================== CONFIG ========================================
N_ENVS = min(8, max(2, multiprocessing.cpu_count() - 1))   # parallel sims
STEPS  = 2_000_000          # total timesteps (raise to 12_000_000 for full run)
OUT    = 'experiments/rir/rir_notch_NEW'    # where the model is saved
RUN_NAME = 'RIR_NOTCH_NEW'
VALUE_WARMUP = 1_000_000    # critic warm-up steps before policy updates
BC_START, BC_END = 1.0, 0.3 # behaviour-cloning anchor weight (DAPG), decays
# ==============================================================================

cmd = [
    sys.executable, '-m', 'experiments.rir.train_rir',
    '--bc',   'experiments/rir/bc_model_notch2',
    '--data', 'experiments/rir/bt_dataset_notch2.npz',
    '--red_doctrine',
    '--bc_start', str(BC_START), '--bc_end', str(BC_END),
    '--n_envs', str(N_ENVS),
    '--steps',  str(STEPS),
    '--value_warmup', str(VALUE_WARMUP),
    '--out', OUT, '--run_name', RUN_NAME,
    '--device', 'cpu',
]

env = dict(os.environ)
env['PYTHONPATH'] = HERE + os.pathsep + env.get('PYTHONPATH', '')
env['PYTHONUTF8'] = '1'                 # UTF-8 mode in the training subprocess
env['PYTHONIOENCODING'] = 'utf-8'       # and in its spawned env workers

print("=" * 70)
print("TRAINING BLUE AGENT")
print(f"  parallel envs : {N_ENVS}")
print(f"  total steps   : {STEPS:,}")
print(f"  output model  : {OUT}")
print("  (close the Run panel or stop the process to abort)")
print("=" * 70)
print("command:", " ".join(cmd), "\n", flush=True)

# run as a subprocess so Python multiprocessing (spawn on Windows) works cleanly
sys.exit(subprocess.call(cmd, cwd=HERE, env=env))
