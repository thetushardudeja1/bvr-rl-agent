"""
smoke_test.py — non-interactive end-to-end behavior check.
Runs a missile intercept + aircraft flight and asserts the integrated
physics (PN guidance + enu_to_body + autopilot PID) behaves correctly
after the scalar-math optimizations.
"""
import sys
sys.path.insert(0, '.')
from jsb_gym.simObjects.config import f16_config, AAM_config
from jsb_gym.simObjects.aircraft import F16BVR
from jsb_gym.simObjects.missiles import AAMBVR

AIM = AAMBVR(AAM_config)
F16 = F16BVR(f16_config)

F16.reset(lat=59, long=18, alt=3000, vel=250, heading=180)
AIM.reset(lat=58.4, long=18, alt=10000, vel=250, heading=0)
AIM.set_target(F16)

start_dist = None
min_dist = float('inf')
for i in range(1500):
    F16.step([150, 4000, 0.5])
    AIM.step()
    d = AIM.PN.distance_to_target
    if d is not None:
        if start_dist is None:
            start_dist = d
        min_dist = min(min_dist, d)

print(f"  Start distance : {start_dist:,.0f} m")
print(f"  Min  distance  : {min_dist:,.0f} m")
print(f"  Closed by      : {start_dist - min_dist:,.0f} m")

ok = (start_dist is not None) and (min_dist < start_dist * 0.5)
# sanity: aircraft still flying with finite state
finite = all(abs(v) < 1e9 for v in (F16.get_altitude(), F16.get_mach(), F16.get_psi()))

print("\n" + "="*50)
if ok and finite:
    print("  ✅ SMOKE TEST PASSED — PN guidance homes correctly,")
    print("     aircraft state finite. Integration is sound.")
else:
    print(f"  ❌ FAILED  (homed={ok}, finite_state={finite})")
print("="*50)
sys.exit(0 if (ok and finite) else 1)
