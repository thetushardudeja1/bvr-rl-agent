"""
gen_missile_envelope.py — show each in-flight MISSILE's reachable footprint in
Tacview: a ring drawn around the missile whose radius = how far it can still fly
given its current Mach (from experiments/wez/flyout.json). The bubble SHRINKS as
the missile bleeds energy, and collapses when it is dragged/notched -> you watch
the threat literally run out of reach.

Ring colour = energy band:  Red (Mach>=2.5) hot / Orange (1.6-2.5) / Yellow (<1.6).
A thin cyan R_max ring is also drawn around the shooter for context.

Run:  python gen_missile_envelope.py [seed]
"""
import sys, os, json, math, shutil, argparse
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.loggers import TacviewLogger

ap = argparse.ArgumentParser()
ap.add_argument('seed', nargs='?', type=int, default=7529)
ap.add_argument('--model', default='experiments/rir/rir_notch2_best')
ap.add_argument('--npts', type=int, default=40)
A = ap.parse_args()
OUT = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/wez_tacview'
os.makedirs(OUT, exist_ok=True)

fly = json.load(open('experiments/wez/flyout.json'))
TAB = {float(k): float(v) for k, v in fly['remaining_by_mach'].items()}
MACHS = sorted(TAB)

def remaining_km(mach):
    if mach <= MACHS[0]:  return TAB[MACHS[0]]
    if mach >= MACHS[-1]: return TAB[MACHS[-1]]
    for i in range(1, len(MACHS)):
        if mach <= MACHS[i]:
            a, b = MACHS[i-1], MACHS[i]
            t = (mach - a) / (b - a)
            return TAB[a] + t * (TAB[b] - TAB[a])
    return TAB[MACHS[-1]]

def band(mach):
    return 'Red' if mach >= 2.5 else 'Orange' if mach >= 1.6 else 'Yellow'

def offset(lat, lon, brg, dist_m):
    b = math.radians(brg)
    return (lat + (dist_m * math.cos(b)) / 111_320.0,
            lon + (dist_m * math.sin(b)) / (111_320.0 * math.cos(math.radians(lat))))

baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(A.model, device='cpu')
env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=A.seed)
env.tacview_logger = TacviewLogger(env); lg = env.tacview_logger
BASE = 0xA00000

def bubble(missile, slot_idx):
    try:
        lat = missile.get_lat_gc_deg(); lon = missile.get_long_gc_deg()
        alt = missile.get_altitude();   mach = missile.get_mach()
    except Exception:
        return
    rk = remaining_km(mach); col = band(mach)
    base = BASE + slot_idx * (A.npts + 1)
    for i in range(A.npts):
        a = 360.0 * i / A.npts
        mlat, mlon = offset(lat, lon, a, rk * 1e3)
        lg.lines.append(f"{base+i:X},T={mlon:.6f}|{mlat:.6f}|{alt:.0f},"
                        f"Type=Misc+Shrapnel,Color={col}")

def draw(frame):
    slot = 0
    for side in (env.red_agent, env.blue_agent):
        for k, m in side.ammo.items():
            if m.is_active() or getattr(m, 'coasting', False):
                bubble(m, slot)
            slot += 1

done = trunc = False; frame = 0
while not (done or trunc):
    lg.log_flight_data(); draw(frame); frame += 1
    act, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, info = env.step(act)
for _ in range(8):
    lg.log_flight_data(); draw(frame); frame += 1
    act, _ = model.predict(obs, deterministic=True)
    try: obs, _, _, _, info = env.step(act)
    except Exception: break
lg.save_logs()

name = f"missileENV_seed{A.seed}_{info.get('outcome','?')}.acmi"
shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
            os.path.join(OUT, name))
print(f"SAVED: {os.path.join(OUT, name)}")
print(f"  flyout table: Mach {MACHS[0]:.2f}->{remaining_km(MACHS[0]):.0f}km .. "
      f"Mach {MACHS[-1]:.2f}->{remaining_km(MACHS[-1]):.0f}km")
