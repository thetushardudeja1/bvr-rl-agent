"""
================================================================================
 RUN_eval.py  --  PRESS ▶ "RUN" TO MEASURE THE AGENT'S WIN RATE
================================================================================
No terminal needed. Open in VS Code, click Run (▶) or F5.

Plays the trained Blue agent vs the Red behaviour-tree for N episodes and prints
the win rate plus a breakdown of HOW Blue defeated red's missiles (notch/beam vs
kinematic drag) and which evasion repertoire it used.

Output -> printed in the VS Code Run panel.
================================================================================
"""
import os, sys, subprocess
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))

# ============================== CONFIG ========================================
MODEL    = 'experiments/rir/rir_notch2_best'
EPISODES = 40
# ==============================================================================

env = dict(os.environ); env['PYTHONPATH'] = HERE + os.pathsep + env.get('PYTHONPATH', '')
env['PYTHONUTF8'] = '1'; env['PYTHONIOENCODING'] = 'utf-8'
cmd = [sys.executable, 'extra_scripts/eval_notch.py', '--model', MODEL, '--eps', str(EPISODES)]
print(f"Evaluating {MODEL} over {EPISODES} episodes ...\n", flush=True)
sys.exit(subprocess.call(cmd, cwd=HERE, env=env))
