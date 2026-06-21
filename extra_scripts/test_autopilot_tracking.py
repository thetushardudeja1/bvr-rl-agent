"""
test_autopilot_tracking.py — verify the rebuilt proportional-bank autopilot:
  1. tracks a 90 deg heading change without corkscrew oscillation
  2. holds altitude through the turn
  3. settles (no limit cycle) — bank stays low once on heading
"""
import sys, math
sys.path.insert(0, '.')
from jsb_gym.simObjects.config import f16_config
from jsb_gym.simObjects.aircraft import F16BVR
from jsb_gym.utils.navigation import delta_heading

F16 = F16BVR(f16_config)
F16.reset(lat=59, long=18, alt=8000, vel=250, heading=0)

TARGET_HEAD, TARGET_ALT = 90.0, 8000.0
heads, phis, alts = [], [], []
for i in range(240):  # 240 agent steps x 0.5 s = 120 s
    F16.step([TARGET_HEAD, TARGET_ALT, 0.6])
    heads.append(F16.get_psi())
    phis.append(F16.get_phi())
    alts.append(F16.get_altitude())

# metrics over the last 60 s (settled regime)
tail_err  = [abs(delta_heading(TARGET_HEAD, h)) for h in heads[120:]]
tail_phi  = [abs(p) for p in phis[120:]]
tail_alt  = [abs(a - TARGET_ALT) for a in alts[120:]]
# overshoot: worst heading past the target during the whole run
signed = [delta_heading(TARGET_HEAD, h) for h in heads]
overshoot = max(0.0, -min(signed))

print(f"settled heading err : mean {sum(tail_err)/len(tail_err):6.2f} deg  max {max(tail_err):6.2f}")
print(f"settled |bank|      : mean {sum(tail_phi)/len(tail_phi):6.2f} deg  max {max(tail_phi):6.2f}")
print(f"settled alt err     : mean {sum(tail_alt)/len(tail_alt):6.1f} m    max {max(tail_alt):6.1f}")
print(f"heading overshoot   : {overshoot:.2f} deg")

ok = (sum(tail_err)/len(tail_err) < 5.0 and
      max(tail_phi) < 30.0 and          # no corkscrew: bank stays small once settled
      sum(tail_alt)/len(tail_alt) < 300.0 and
      overshoot < 25.0)
print("\n" + ("✅ TRACKING TEST PASSED" if ok else "❌ TRACKING TEST FAILED"))
sys.exit(0 if ok else 1)
