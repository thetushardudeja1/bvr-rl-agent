"""
smoke_notch.py — validate the Doppler-notch sensor layer + new 19-dim obs.

Test 1 (UNIT): target_los_speed geometry on synthetic positions/velocities.
  Missile at origin, target 10 km NORTH (horizontal LOS points north).
    - target flying NORTH  (along LOS)        -> along-LOS ~= |v|   (closing/drag)
    - target flying EAST   (perp / beaming)   -> along-LOS ~= 0      (notch)
    - target flying NE 45  (half)             -> along-LOS ~= |v|/sqrt2
Test 2 (INTEGRATION): env builds, obs shape (40,19), finite, steps; and an
  ordinary head-on closing run does NOT spuriously notch the missile.
"""
import sys, numpy as np
sys.path.insert(0, '.')
from jsb_gym.simObjects.missiles import AAMBVR

class Fake:
    def __init__(self, lat, lon, alt, ve=0.0, vn=0.0):
        self.lat, self.lon, self.alt, self.ve, self.vn = lat, lon, alt, ve, vn
    def get_lat_gc_deg(self): return self.lat
    def get_long_gc_deg(self): return self.lon
    def get_altitude(self):   return self.alt
    def get_v_east(self):     return self.ve
    def get_v_north(self):    return self.vn

DEG_PER_M = 1.0 / 111320.0
missile = Fake(0.0, 0.0, 8000.0)                 # acts as 'self'
def los(ve, vn):
    missile.target = Fake(10000*DEG_PER_M, 0.0, 8000.0, ve=ve, vn=vn)  # 10km north
    return AAMBVR.target_los_speed(missile)

print("=== Test 1: target_los_speed geometry (unit) ===")
v = 300.0
along = los(0.0,  v)      # north = along LOS
beam  = los(v,    0.0)    # east  = perpendicular (beam)
diag  = los(v/np.sqrt(2), v/np.sqrt(2))
print(f"  along-LOS (north fly): {along:6.1f}  (expect ~{v})")
print(f"  beam     (east  fly): {beam:6.1f}  (expect ~0)")
print(f"  diagonal (NE 45 fly): {diag:6.1f}  (expect ~{v/np.sqrt(2):.0f})")
ok1 = abs(along-v) < 5 and beam < 5 and abs(diag - v/np.sqrt(2)) < 5
print(f"  -> {'PASS ✅' if ok1 else 'FAIL ❌'}")

print("\n=== Test 2: env obs shape + no spurious notch on head-on close ===")
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.scale import scale_between
env = BVRBase(baseEnv_conf); env.reset(seed=7)
assert env.state.shape == (40, 19), f"bad obs shape {env.state.shape}"
assert np.isfinite(env.state).all(), "non-finite obs"
hold_psi = env.observation['bearing'] % 360          # fix heading once: NO turning
def act_const():
    a = np.zeros(4, np.float32)
    a[0] = scale_between(hold_psi, 0, 360)
    a[1] = scale_between(8000.0, 3e3, 12e3); a[2] = 1.0; a[3] = -1.0
    return a
spurious = False; steps = 0; done = trunc = False
while not (done or trunc) and steps < 1800:
    _, _, done, trunc, info = env.step(act_const())
    for m in env.red_agent.ammo.values():
        if m.target_lost and m.notch_count > m.missile_performance.notch_count:
            spurious = True
    steps += 1
hit = env.blue_agent.healthPoints <= 0.0
print(f"  obs shape (40,19): PASS ✅   finite: PASS ✅")
print(f"  straight (constant-heading) target notched the missile: "
      f"{'YES (bad) ❌' if spurious else 'NO ✅'}")
print(f"  straight target was hit by missile: {'YES ✅' if hit else 'no'}"
      f"   outcome={info.get('outcome','?')}")
print(f"\nOVERALL: {'ALL PASS ✅' if (ok1 and not spurious) else 'CHECK ABOVE ❌'}")
