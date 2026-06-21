"""BTvsBT sanity under the new WEZ gate + randomized ICs: do engagements
still resolve (kills happen, episodes terminate)?"""
import sys
sys.path.insert(0, '.')
from jsb_gym.envs.BaseEnv import BVREnv_BTvsBT
from jsb_gym.envs.config import baseEnv_conf

outcomes = []
for ep in range(6):
    env = BVREnv_BTvsBT(baseEnv_conf)
    env.reset(seed=ep)
    done = trunc = False
    steps = 0
    while not (done or trunc):
        _, _, done, trunc, info = env.step(None)
        steps += 1
    outcomes.append((info.get('outcome', '?'), steps))
    print(f"  ep {ep}: {outcomes[-1][0]:8s} in {steps} steps")

kills = sum(1 for o, _ in outcomes if o in ('win', 'loss'))
print(f"\n  kills: {kills}/6, timeouts: {6-kills}/6")
ok = kills >= 2  # symmetric BTs with WEZ gate: expect a decent kill share
print("✅ BTvsBT OUTCOME TEST PASSED" if ok else "❌ FAILED — too few kills, WEZ gate may be over-restrictive")
sys.exit(0 if ok else 1)
