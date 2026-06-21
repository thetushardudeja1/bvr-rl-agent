"""
Ground-track tracker: logs ENU positions of blue, red, and every red missile
each env step, so we can see whether a NOTCHED (coasting) red missile actually
flies back toward red. Prints, per red missile, the distance-to-red over time
and the minimum distance to red reached AFTER lock loss.

Run:  python missile_track.py [seed]
"""
import sys, os
sys.path.insert(0, '.')
import pymap3d as pm
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf

SEED  = int(sys.argv[1]) if len(sys.argv) > 1 else 7668
MODEL = 'experiments/rir/rir_notch2_best'
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')
env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=SEED)

def pos(o):
    try:    return o.simObj.get_lat_gc_deg(), o.simObj.get_long_gc_deg(), o.simObj.get_altitude()
    except: return o.get_lat_gc_deg(), o.get_long_gc_deg(), o.get_altitude()

# origin = blue start
olat, olon, oalt = pos(env.blue_agent)
def enu(o):
    la, lo, al = pos(o)
    e, n, u = pm.geodetic2enu(la, lo, al, olat, olon, oalt)
    return e, n, u

tracks = {}   # mid -> list of (step, dist_to_red, dist_to_blue, coasting, alive, notch_count)
step = 0
done = trunc = False
while not (done or trunc):
    a, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, info = env.step(a)
    step += 1
    be, bn, bu = enu(env.blue_agent)
    re, rn, ru = enu(env.red_agent)
    for k, m in env.red_agent.ammo.items():
        if not (m.is_active() or getattr(m, 'coasting', False) or id(m) in tracks):
            continue
        me, mn, mu = enu(m)
        d_red  = ((me-re)**2 + (mn-rn)**2 + (mu-ru)**2) ** 0.5
        d_blue = ((me-be)**2 + (mn-bn)**2 + (mu-bu)**2) ** 0.5
        tracks.setdefault(id(m), []).append(
            (step, d_red, d_blue, int(getattr(m, 'coasting', False)),
             int(m.alive), m.notch_count))

print(f"\nseed {SEED}  outcome={info.get('outcome','?')}\n")
for n, (mid, t) in enumerate(tracks.items(), 1):
    # find first coasting frame
    coast_idx = next((i for i, r in enumerate(t) if r[3] == 1), None)
    notched = any(r[5] > env.red_agent.ammo['0'].missile_performance.notch_count for r in t)
    print(f"--- red missile #{n}  ({len(t)} steps, notched={notched}) ---")
    if coast_idx is not None:
        post = t[coast_idx:]
        d_red_at_notch = t[coast_idx][1]
        min_d_red_after = min(r[1] for r in post)
        end_d_red = post[-1][1]
        print(f"  dist to RED at lock-loss : {d_red_at_notch/1e3:6.2f} km")
        print(f"  MIN dist to RED after    : {min_d_red_after/1e3:6.2f} km   <-- did it come back?")
        print(f"  dist to RED at end       : {end_d_red/1e3:6.2f} km")
        # print a coarse trace
        print("  trace (step: d_red km / d_blue km):")
        for r in post[::max(1, len(post)//12)]:
            print(f"    {r[0]:4d}:  red {r[1]/1e3:6.2f}   blue {r[2]/1e3:6.2f}   {'COAST' if r[3] else ''}{' DEAD' if not r[4] else ''}")
    else:
        print("  (never coasted — hit or other outcome)")
    print()
