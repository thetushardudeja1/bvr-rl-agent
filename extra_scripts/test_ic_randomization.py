"""Verify IC randomization: geometry differs across resets, within ranges."""
import sys
sys.path.insert(0, '.')
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents

env = BVRBase(baseEnv_conf)
dists, alts, heads = [], [], []
for i in range(5):
    env.reset(seed=100 + i)
    d = dinstance_between_agents(env.blue_agent, env.red_agent)
    dists.append(d)
    alts.append(env.blue_agent.simObj.get_altitude())
    heads.append(env.blue_agent.simObj.get_psi())
    print(f"  reset {i}: range={d/1e3:6.1f}km  blue_alt={alts[-1]:7.0f}m  blue_head={heads[-1]:6.1f}")

lo, hi = baseEnv_conf.ic_ranges['range_m']
ok = (len(set(round(d) for d in dists)) == 5 and
      all(lo*0.95 <= d <= hi*1.05 for d in dists) and
      len(set(round(h) for h in heads)) >= 4)
print("\n" + ("✅ IC RANDOMIZATION TEST PASSED" if ok else "❌ FAILED"))
sys.exit(0 if ok else 1)
