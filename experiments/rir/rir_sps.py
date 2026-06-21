"""Measure PPO+AEG training SPS at the real config."""
import sys, time, copy, multiprocessing
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.envs.vec_env import ChunkSubprocVecEnv
from experiments.rir.ppo_aeg import PPO_AEG

N_ENVS, N_STEPS, BATCH = 360, 128, 8192
TOTAL = N_STEPS * N_ENVS * 3  # 3 rollouts

def main():
    env = make_vec_env(BVRBase, n_envs=N_ENVS, vec_env_cls=ChunkSubprocVecEnv,
                       env_kwargs={'conf': baseEnv_conf})
    m = PPO_AEG("MlpPolicy", env, device='cpu', verbose=0,
                n_steps=N_STEPS, batch_size=BATCH, ent_coef=0.01,
                aeg_coef_start=1.0, aeg_coef_end=0.0)
    m.reference_policy = copy.deepcopy(m.policy)
    for p in m.reference_policy.parameters(): p.requires_grad_(False)
    m.reference_policy.eval()
    t = time.time()
    m.learn(total_timesteps=TOTAL)
    el = time.time() - t
    sps = int(TOTAL / el)
    print(f"\n  RIR (PPO+AEG) TRAINING SPS: {sps:,}  (~{(15_000_000/sps)/3600:.1f} hrs for 15M)")
    env.close()

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
