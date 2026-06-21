import sys, os
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf

MODEL = 'experiments/rir/rir_notch2_best'
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')

wins = 0
losses = 0
timeouts = 0
notches = 0

print("Running 20 matches to test Blue's performance against FIXED missiles...")
for seed in range(7500, 7520):
    env = BVRBase(baseEnv_conf)
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    
    seen = {}
    match_notches = 0
    
    while not (done or trunc):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        
        for m in env.red_agent.ammo.values():
            if m.target_lost and id(m) not in seen:
                if m.notch_count > m.missile_performance.notch_count:
                    seen[id(m)] = 'n'
                    match_notches += 1
                else:
                    seen[id(m)] = 'k'
                    
    oc = info.get('outcome', '?')
    if oc == 'win': wins += 1
    elif oc == 'loss': losses += 1
    else: timeouts += 1
    notches += match_notches
    
    print(f"Seed {seed}: {oc.upper()} | Notches: {match_notches}")

print(f"\nRESULTS OUT OF 20 MATCHES:")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Timeouts: {timeouts}")
print(f"Total successful notches: {notches}")
if wins == 0:
    print("\nCONCLUSION: Blue agent cannot win against the newly fixed missiles! Retraining is definitely required.")
