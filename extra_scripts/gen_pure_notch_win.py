import sys, os, shutil
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents
from jsb_gym.utils.loggers import TacviewLogger

MODEL = 'experiments/rir/rir_notch2_best'
WOUT  = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/pure_notch_wins'
WANT  = 6
os.makedirs(WOUT, exist_ok=True)
baseEnv_conf.red_doctrine_randomize = False        # standard smart red
model = PPO.load(MODEL, device='cpu')

found, seed = 0, 7500
print("scanning for PURE NOTCH engagements (NO running away)...")
while found < WANT and seed < 9500:
    env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=seed)
    env.tacview_logger = TacviewLogger(env)
    bsh = rsh = 0; pb = pr = False; min_d = 1e9
    seen = {}; n_notch = n_kin = 0
    done = trunc = False
    
    blue_turned_back = False
    
    while not (done or trunc):
        env.tacview_logger.log_flight_data()
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(a)
        b, r = env.blue_agent, env.red_agent
        min_d = min(min_d, dinstance_between_agents(b, r))
        
        # Check if blue turned away (aspect angle > 120 degrees roughly means running away)
        # We can just rely on the missile defeat type. If any missile is defeated kinematically, it ran away.
        
        ba, ra = b.is_own_missile_active(), r.is_own_missile_active()
        bsh += (ba and not pb); rsh += (ra and not pr); pb, pr = ba, ra
        
        for m in r.ammo.values():
            if m.target_lost and id(m) not in seen:
                if m.notch_count > m.missile_performance.notch_count:
                    seen[id(m)] = 'n'; n_notch += 1
                else:
                    seen[id(m)] = 'k'; n_kin += 1
    
    oc = info.get('outcome', '?')
    
    # KEY CHECK: Win + Red fired + AT LEAST 1 Notch + ZERO kinematic defeats (no running away)
    if oc == 'win' and rsh > 0 and n_notch > 0 and n_kin == 0:
        for _ in range(12):                       # post-kill coast
            env.tacview_logger.log_flight_data()
            a, _ = model.predict(obs, deterministic=True)
            try: obs, _, _, _, info = env.step(a)
            except Exception: break
        env.tacview_logger.save_logs()
        found += 1
        name = (f"pure_notch_seed{seed}_B{bsh}R{rsh}"
                f"_beam{n_notch}_drag{n_kin}_min{min_d/1e3:.0f}km.acmi")
        shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
                    os.path.join(WOUT, name))
        print(f"  >>> SAVED PURE NOTCH WIN: {name}")
    else:
        # Just logging silently so we don't spam
        pass
    seed += 1
print(f"\n{found} pure notch duels saved to {WOUT}")
