"""
diagnose_timeouts.py
====================
Tests the hypothesis that timeouts are caused by WASTED missiles (fired at bad
range, missed, out of ammo, enemy alive) rather than failure to engage.

For each episode, classifies every missile as: hit / lost(missed) / never-fired,
and records the range at which each was fired. Aggregates by outcome.
"""
import sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents

N = 50
model = PPO.load('experiments/rir/rir_model_best', device='cpu')
stats = defaultdict(list)

for ep in range(N):
    env = BVRBase(baseEnv_conf)
    obs, _ = env.reset()
    blue = env.blue_agent
    fire_range = {}
    done = trunc = False
    min_dist = 1e9
    info = {}
    while not (done or trunc):
        a, _ = model.predict(obs, deterministic=False)
        obs, _, done, trunc, info = env.step(a)
        d = dinstance_between_agents(blue, env.red_agent)
        min_dist = min(min_dist, d)
        for mid, m in blue.ammo.items():
            if m.is_active() and mid not in fire_range:
                fire_range[mid] = d
    oc = info.get('outcome', '?')
    n_total = len(blue.ammo)
    n_fired = len(fire_range)
    n_hit  = sum(1 for m in blue.ammo.values() if m.is_target_hit())
    n_lost = sum(1 for m in blue.ammo.values() if m.is_traget_lost())
    n_ready = sum(1 for m in blue.ammo.values() if m.is_ready_to_launch())
    stats[oc].append(dict(n_fired=n_fired, n_total=n_total, n_hit=n_hit,
                          n_lost=n_lost, n_ready=n_ready,
                          fr=list(fire_range.values()), md=min_dist))
    print(f"ep{ep:>2} {oc:>8}: fired {n_fired}/{n_total} hit={n_hit} lost={n_lost} "
          f"unfired={n_ready} mind={min_dist/1000:.0f}km "
          f"ranges={[f'{r/1000:.0f}km' for r in fire_range.values()]}")

print("\n" + "=" * 60)
for oc in ['timeout', 'win', 'loss']:
    eps = stats.get(oc, [])
    if not eps:
        continue
    tot = eps[0]['n_total']
    allr = [r for e in eps for r in e['fr']]
    out_of_ammo = [e for e in eps if e['n_ready'] == 0]
    print(f"  {oc.upper()} ({len(eps)} eps):")
    print(f"     avg fired      : {np.mean([e['n_fired'] for e in eps]):.1f}/{tot}")
    print(f"     avg missed(lost): {np.mean([e['n_lost'] for e in eps]):.1f}")
    print(f"     avg never-fired : {np.mean([e['n_ready'] for e in eps]):.1f}")
    print(f"     out-of-ammo     : {len(out_of_ammo)}/{len(eps)} ({100*len(out_of_ammo)/len(eps):.0f}%)")
    if allr:
        print(f"     avg fire range  : {np.mean(allr)/1000:.0f} km")
    print(f"     avg min dist    : {np.mean([e['md'] for e in eps])/1000:.0f} km")
print("=" * 60)
