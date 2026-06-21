#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bvrc
cd ~/bvrgym_project_new

# Force single-threaded math libs in ALL forked workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MAX_RESTARTS=10
restarts=0

echo "=== Auto-Resume Watchdog Started ==="

while [ $restarts -lt $MAX_RESTARTS ]; do
    echo "Starting training run (Restart $restarts of $MAX_RESTARTS)..."
    
    python experiments/v2/train_selfplay_v2.py
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training finished successfully! Target generations reached."
        break
    else
        echo "WARNING: Training crashed or was killed with exit code $EXIT_CODE."
        echo "Restarting in 10 seconds..."
        restarts=$((restarts+1))
        sleep 10
    fi
done

if [ $restarts -eq $MAX_RESTARTS ]; then
    echo "FATAL: Maximum restarts ($MAX_RESTARTS) reached. Training aborted."
fi
