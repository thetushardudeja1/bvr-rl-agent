"""
Track BLUE missiles relative to RED to test the 'missile rotates around red'
observation. For each blue missile we log distance-to-red and the BEARING from
red to the missile; the total signed sweep of that bearing tells us if the
missile actually circles red (|sweep| >= ~300 deg = an orbit).

Run:  python blue_missile_track.py [seed]
"""
import sys, math
sys.path.insert(0, '.')
import pymap3d as pm
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7578
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load('experiments/rir/rir_notch2_best', device='cpu')
env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=SEED)

def pos(o):
    try:    return o.simObj.get_lat_gc_deg(), o.simObj.get_long_gc_deg(), o.simObj.get_altitude()
    except: return o.get_lat_gc_deg(), o.get_long_gc_deg(), o.get_altitude()

tracks = {}
step = 0; done = trunc = False
while not (done or trunc):
    a, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, info = env.step(a)
    step += 1
    rlat, rlon, ralt = pos(env.red_agent)
    for k, m in env.blue_agent.ammo.items():
        if not (m.is_active() or getattr(m, 'coasting', False) or id(m) in tracks):
            continue
        mlat, mlon, malt = pos(m)
        e, n, _ = pm.geodetic2enu(mlat, mlon, malt, rlat, rlon, ralt)  # missile rel to red
        brg = math.degrees(math.atan2(e, n)) % 360.0
        d = (e*e + n*n) ** 0.5
        tracks.setdefault(id(m), []).append(
            (step, d, brg, int(getattr(m, 'coasting', False)), int(m.alive),
             int(m.target_hit)))

print(f"\nseed {SEED}  outcome={info.get('outcome','?')}\n")
for n, (mid, t) in enumerate(tracks.items(), 1):
    # signed bearing sweep
    sweep = 0.0
    for i in range(1, len(t)):
        d = t[i][2] - t[i-1][2]
        while d > 180: d -= 360
        while d < -180: d += 360
        sweep += d
    dmin = min(r[1] for r in t)
    ended = 'HIT' if t[-1][5] else ('coast/lost' if t[-1][3] or not t[-1][4] else 'flying')
    print(f"--- blue missile #{n}: {len(t)} steps, min dist to red {dmin:.0f} m, "
          f"end={ended} ---")
    print(f"    total bearing sweep around red = {sweep:.0f} deg  "
          f"{'>>> ORBITS RED' if abs(sweep) > 300 else '(no full orbit)'}")
    # show the close phase (within 2 km of red)
    near = [r for r in t if r[1] < 2000]
    if near:
        print("    near-red frames (step: dist m, bearing deg, alive, hit):")
        for r in near[::max(1, len(near)//10)]:
            print(f"      {r[0]:4d}:  {r[1]:6.0f} m   brg {r[2]:6.1f}   "
                  f"alive={r[4]} hit={r[5]} coast={r[3]}")
