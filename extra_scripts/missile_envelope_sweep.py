"""
missile_envelope_sweep.py
=========================
WEZ characterization: fire the AAM across a grid of launch range x target
aspect x altitude against a straight-flying (non-evading) F-16 and record
hit/miss. Produces:
  - experiments/wez/envelope.json   (raw results + max-hit-range table)
  - experiments/wez/envelope.png    (heatmap)

Aspect convention: 180 = head-on (target flying AT the launcher),
90 = beam, 0 = tail-chase (target flying away).
"""
import sys, os, json, math, time
from types import SimpleNamespace
sys.path.insert(0, '.')
from jsb_gym.simObjects.config import f16_config, AAM_config
from jsb_gym.simObjects.aircraft import F16BVR
from jsb_gym.simObjects.missiles import AAMBVR

RANGES_KM = [10, 20, 30, 40, 50, 60, 70, 80]
ASPECTS   = [180, 90, 0]          # head-on, beam, tail-chase
ALTS_M    = [5e3, 8e3, 11e3]      # both aircraft co-altitude
SHOOTER_SPEEDS = [250.0, 330.0, 400.0]   # m/s — launch energy axis
TGT_SPEED = 330.0                 # m/s
MAX_SIM_S = 180.0

def fire_one(range_m, aspect_deg, alt_m, shooter, target_ac, missile, shooter_vel=330.0):
    lat0, lon0 = 58.0, 18.0
    # target placed due north of shooter; LOS bearing = 0
    tgt_lat = lat0 + range_m / 111_320.0
    # target heading: 180 flies south (head-on), 0 flies north (away)
    tgt_head = (180.0 + (180.0 - aspect_deg)) % 360.0

    shooter.reset(lat=lat0, long=lon0, alt=alt_m, vel=shooter_vel, heading=0.0)
    target_ac.reset(lat=tgt_lat, long=lon0, alt=alt_m, vel=TGT_SPEED, heading=tgt_head)

    missile.reset_operational_state()
    target = SimpleNamespace(simObj=target_ac, healthPoints=1.0)
    missile.set_target(target)
    missile.launch(SimpleNamespace(simObj=shooter))

    t, min_d = 0.0, float('inf')
    while missile.is_active() and t < MAX_SIM_S:
        target_ac.step([tgt_head, alt_m, 0.5])   # straight, non-evading
        missile.step()
        if missile.target_norm is not None:
            min_d = min(min_d, missile.target_norm)
        t += 0.5
    return missile.target_hit, min_d, t

def main():
    os.makedirs('experiments/wez', exist_ok=True)
    shooter   = F16BVR(f16_config)
    target_ac = F16BVR(f16_config)
    missile   = AAMBVR(AAM_config)

    results = []
    t0 = time.time()
    for vel in SHOOTER_SPEEDS:
        for alt in ALTS_M:
            for asp in ASPECTS:
                for rng_km in RANGES_KM:
                    hit, min_d, dur = fire_one(rng_km*1e3, asp, alt, shooter,
                                               target_ac, missile, shooter_vel=vel)
                    results.append(dict(range_km=rng_km, aspect=asp, alt_m=alt,
                                        shooter_vel=vel, hit=bool(hit),
                                        miss_dist_m=round(min_d), dur_s=dur))
                    print(f"  v={vel:.0f} alt={alt/1e3:.0f}km asp={asp:>3} R={rng_km:>2}km -> "
                          f"{'HIT ' if hit else 'MISS'} (closest {min_d:,.0f} m, {dur:.0f}s)")

    # max hit range per (aspect, alt, shooter speed)
    rmax = {}
    for asp in ASPECTS:
        for alt in ALTS_M:
            for vel in SHOOTER_SPEEDS:
                hits = [r['range_km'] for r in results
                        if r['aspect'] == asp and r['alt_m'] == alt
                        and r['shooter_vel'] == vel and r['hit']]
                rmax[f"asp{asp}_alt{int(alt/1e3)}km_v{int(vel)}"] = max(hits) if hits else 0

    with open('experiments/wez/envelope.json', 'w') as f:
        json.dump({'results': results, 'rmax_km': rmax,
                   'note': 'non-evading target, co-altitude, straight flight'}, f, indent=1)

    print(f"\n  Max hit range (km) by aspect/alt: {json.dumps(rmax, indent=2)}")
    print(f"  Total {len(results)} shots in {time.time()-t0:.0f}s")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, alt in zip(axes, ALTS_M):
            grid = [[1 if next(r['hit'] for r in results
                     if r['range_km']==rk and r['aspect']==a and r['alt_m']==alt
                     and r['shooter_vel']==330.0) else 0
                     for rk in RANGES_KM] for a in ASPECTS]
            ax.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(len(RANGES_KM)), RANGES_KM)
            ax.set_yticks(range(len(ASPECTS)), [f"{a}°" for a in ASPECTS])
            ax.set_xlabel('launch range (km)'); ax.set_title(f'alt {alt/1e3:.0f} km')
        axes[0].set_ylabel('target aspect')
        fig.suptitle('Measured missile WEZ (green=hit) — non-evading target')
        fig.tight_layout()
        fig.savefig('experiments/wez/envelope.png', dpi=120)
        print("  chart -> experiments/wez/envelope.png")
    except Exception as e:
        print(f"  (no chart: {e})")

if __name__ == '__main__':
    main()
