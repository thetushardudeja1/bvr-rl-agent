"""
train_sps_bench_v2.py
=====================
Same methodology as train_sps_bench_v3.py but on the V2 self-play env
(one 0.5s base step per agent decision). Directly comparable numbers.
"""
import sys, time, multiprocessing
sys.path.insert(0, '.')
import torch
torch.set_num_threads(1)

from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.envs.vec_env import ChunkSubprocVecEnv
from experiments.v2.selfplay_env_v2 import SelfPlayBVR
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

N_ENVS  = 300
N_STEPS = 128
BATCH   = 8192
ROLLOUT = N_ENVS * N_STEPS
TOTAL   = ROLLOUT * 2

def main():
    baseEnv_conf.opponent_path = 'BT'
    vec_env = make_vec_env(SelfPlayBVR, n_envs=N_ENVS,
                           vec_env_cls=ChunkSubprocVecEnv,
                           env_kwargs={'conf': baseEnv_conf})
    model = PPO("MlpPolicy", vec_env, verbose=0, device='cpu',
                learning_rate=5e-5, n_steps=N_STEPS, batch_size=BATCH,
                ent_coef=0.003, gamma=0.999)
    model.learn(total_timesteps=ROLLOUT // 4)   # warmup
    start = time.time()
    model.learn(total_timesteps=TOTAL, reset_num_timesteps=True)
    elapsed = time.time() - start
    sps = TOTAL / elapsed
    print(f"\n  REAL V2 TRAINING SPS : {sps:,.0f} base-steps/s")
    print(f"  sim-seconds/s        : {sps*0.5:,.0f}  (V3 @557 = {557*2:,.0f})")
    vec_env.close()

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
