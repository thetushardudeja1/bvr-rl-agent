"""Make a lightweight, demo-free copy of the seed model so it can be used as a
self-play opponent without loading the 92MB demonstration tensor in every
worker (that memory spike killed the previous run)."""
import sys
sys.path.insert(0, '.')
from experiments.rir.ppo_aeg import PPO_AEG

src = 'experiments/rir/rir_model_best'
dst = 'experiments/rir/seed_clean'
m = PPO_AEG.load(src, device='cpu')
m.demo_obs = None
m.demo_acts = None
m.save(dst)
print(f"  saved demo-free seed -> {dst}.zip")
