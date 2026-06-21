"""
calibrate_missile_flyout.py — measure the missile's REMAINING reachable range as
a function of its current Mach (energy state).

Fire one missile straight at an unreachable (300 km) non-maneuvering target so it
flies until it decelerates below kill-Mach and dies. Log, each step, the current
Mach and how far downrange it has flown. remaining(mach) = total_flyout - flown.
-> experiments/wez/flyout.json  {mach: remaining_km}, monotonic.

Run:  python calibrate_missile_flyout.py [--alt 8000] [--vel 330]
"""
import sys, os, json, argparse
from types import SimpleNamespace
sys.path.insert(0, '.')
import pymap3d as pm
from jsb_gym.simObjects.config import f16_config, AAM_config
from jsb_gym.simObjects.aircraft import F16BVR
from jsb_gym.simObjects.missiles import AAMBVR

ap = argparse.ArgumentParser()
ap.add_argument('--alt', type=float, default=8000.0)
ap.add_argument('--vel', type=float, default=330.0)
A = ap.parse_args()

LAT0, LON0 = 58.0, 18.0
shooter = F16BVR(f16_config); target_ac = F16BVR(f16_config); missile = AAMBVR(AAM_config)

tgt_lat = LAT0 + 300e3 / 111_320.0          # 300 km north -> unreachable
shooter.reset(lat=LAT0, long=LON0, alt=A.alt, vel=A.vel, heading=0.0)
target_ac.reset(lat=tgt_lat, long=LON0, alt=A.alt, vel=330.0, heading=0.0)  # flying away

missile.reset_operational_state()
missile.set_target(SimpleNamespace(simObj=target_ac, healthPoints=1.0))
missile.launch(SimpleNamespace(simObj=shooter))

samples = []   # (mach, flown_km)
t = 0.0
while (missile.is_active() or getattr(missile, 'coasting', False)) and t < 300.0:
    target_ac.step([0.0, A.alt, 0.5])
    missile.step()
    e, n, u = pm.geodetic2enu(missile.get_lat_gc_deg(), missile.get_long_gc_deg(),
                              missile.get_altitude(), LAT0, LON0, A.alt)
    flown_km = ((e*e + n*n) ** 0.5) / 1e3
    samples.append((round(missile.get_mach(), 3), round(flown_km, 3)))
    t += 0.5

total = max(f for _, f in samples) if samples else 0.0
# remaining vs mach: dedup by mach (keep first occurrence on the way down)
table = {}
for mach, flown in samples:
    table.setdefault(mach, total - flown)
table = {m: round(max(0.0, r), 2) for m, r in sorted(table.items())}

os.makedirs('experiments/wez', exist_ok=True)
json.dump({'alt_m': A.alt, 'vel_ms': A.vel, 'total_flyout_km': round(total, 1),
           'remaining_by_mach': table},
          open('experiments/wez/flyout.json', 'w'), indent=1)
print(f"total flyout {total:.1f} km, peak Mach {max(table):.2f}, "
      f"samples {len(samples)} -> experiments/wez/flyout.json")
print("  sample remaining(mach):",
      {k: table[k] for k in list(table)[::max(1, len(table)//8)]})
