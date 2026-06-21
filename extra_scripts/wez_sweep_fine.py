"""
wez_sweep_fine.py — Monte-Carlo Weapon-Employment-Zone characterization.

For each engagement cell (launch range x target aspect x altitude x shooter
speed x target behaviour) we fire N missiles with randomized evasion timing /
target speed / small heading noise, and record the kill fraction -> Pk.

Two target behaviours are swept:
  - 'straight'  : non-maneuvering  -> defines R_max (kinematic reach) and the
                  non-maneuver Pk map.
  - 'evade'     : best-effort evasion (drag or beam, randomized) -> defines
                  R_ne (no-escape) and the evading Pk map.

Outputs (experiments/wez/):
  - wez_pk.json   : full per-cell Pk grid (both behaviours) + metadata
  - wez_rmax_rne.json : derived R_max / R_ne / ratio tables (Pk thresholds)

Aspect convention (matches missile_envelope_sweep.py):
  180 = head-on (target flying AT shooter), 90 = beam, 0 = tail-chase (away).

Run (parallel):  python wez_sweep_fine.py --workers 10 --shots 12
Quick smoke:     python wez_sweep_fine.py --quick --workers 4 --shots 3
"""
import sys, os, json, argparse, random, time
from types import SimpleNamespace
from multiprocessing import Pool
sys.path.insert(0, '.')
from jsb_gym.simObjects.config import f16_config, AAM_config
from jsb_gym.simObjects.aircraft import F16BVR
from jsb_gym.simObjects.missiles import AAMBVR

LAT0, LON0 = 58.0, 18.0
MAX_SIM_S  = 180.0
DT         = 0.5

# ------------------------------------------------------------------ one shot
def fire_one(cell, seed):
    """Run a single stochastic engagement. cell = (range_km, aspect, alt, vel,
    behaviour). Returns 1 (hit) or 0 (miss)."""
    rng_km, aspect, alt, vel, behaviour = cell
    rnd = random.Random(seed)

    # per-shot randomization
    tgt_speed  = 330.0 + rnd.uniform(-25, 25)
    evade_delay = rnd.uniform(0.0, 8.0)                 # s before target reacts
    evade_mode  = rnd.choice(['drag', 'beam'])          # only used if 'evade'
    head_noise  = rnd.uniform(-3, 3)

    shooter   = F16BVR(f16_config)
    target_ac = F16BVR(f16_config)
    missile   = AAMBVR(AAM_config)

    range_m = rng_km * 1e3
    tgt_lat = LAT0 + range_m / 111_320.0               # target due north
    tgt_head = (180.0 + (180.0 - aspect)) % 360.0      # see convention above

    shooter.reset(lat=LAT0, long=LON0, alt=alt, vel=vel, heading=0.0)
    target_ac.reset(lat=tgt_lat, long=LON0, alt=alt, vel=tgt_speed,
                    heading=(tgt_head + head_noise) % 360.0)

    missile.reset_operational_state()
    target = SimpleNamespace(simObj=target_ac, healthPoints=1.0)
    missile.set_target(target)
    missile.launch(SimpleNamespace(simObj=shooter))

    t = 0.0
    while (missile.is_active() or getattr(missile, 'coasting', False)) and t < MAX_SIM_S:
        cmd_head = tgt_head
        if behaviour == 'evade' and t >= evade_delay:
            # drag = run directly away from the shooter (north, heading 0);
            # beam = fly perpendicular to the inbound LOS (east/west).
            cmd_head = 0.0 if evade_mode == 'drag' else 90.0
        target_ac.step([cmd_head % 360.0, alt, 0.5])
        missile.step()
        t += DT
    return 1 if missile.target_hit else 0

def run_cell(args):
    cell, shots, base_seed = args
    hits = sum(fire_one(cell, base_seed + i) for i in range(shots))
    return cell, hits, shots

# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--shots',   type=int, default=12)
    ap.add_argument('--quick',   action='store_true')
    ap.add_argument('--out', default='experiments/wez')
    a = ap.parse_args()

    if a.quick:
        RANGES  = [10, 30, 50]
        ASPECTS = [180, 90, 0]
        ALTS    = [8e3]
        SPEEDS  = [330.0]
    else:
        RANGES  = list(range(5, 71, 5))            # 5..70 km step 5
        ASPECTS = list(range(0, 181, 15))          # 0..180 step 15 (mirror in plot)
        ALTS    = [5e3, 8e3, 11e3]
        SPEEDS  = [250.0, 330.0, 400.0]
    BEHAV = ['straight', 'evade']

    cells = [(r, asp, al, v, b)
             for v in SPEEDS for al in ALTS for asp in ASPECTS
             for b in BEHAV for r in RANGES]
    jobs = [(c, a.shots, 1000 + 97 * i) for i, c in enumerate(cells)]
    total_shots = len(cells) * a.shots
    print(f"cells={len(cells)}  shots/cell={a.shots}  total firings={total_shots}  "
          f"workers={a.workers}")

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    grid = {}
    with Pool(a.workers) as pool:
        for n, (cell, hits, shots) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            r, asp, al, v, b = cell
            key = f"asp{asp}_alt{int(al/1e3)}km_v{int(v)}_{b}"
            grid.setdefault(key, {})[r] = round(hits / shots, 3)
            if n % 25 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} cells  ({el:.0f}s, "
                      f"eta {el/n*(len(jobs)-n):.0f}s)")

    json.dump({'pk': grid, 'shots_per_cell': a.shots,
               'ranges_km': RANGES, 'aspects': ASPECTS,
               'alts_m': ALTS, 'speeds': SPEEDS},
              open(os.path.join(a.out, 'wez_pk.json'), 'w'), indent=1)

    # derive R_max (Pk>=0.5, straight) and R_ne (Pk>=0.5, evade) per geometry
    def max_range_at(prefix, behav, thresh=0.5):
        key = f"{prefix}_{behav}"
        d = grid.get(key, {})
        hit_r = [r for r, pk in d.items() if pk >= thresh]
        return max(hit_r) if hit_r else 0
    rmax, rne, ratio = {}, {}, {}
    for v in SPEEDS:
        for al in ALTS:
            for asp in ASPECTS:
                p = f"asp{asp}_alt{int(al/1e3)}km_v{int(v)}"
                Rm = max_range_at(p, 'straight'); Rn = max_range_at(p, 'evade')
                rmax[p] = Rm; rne[p] = Rn
                ratio[p] = round(Rn / Rm, 2) if Rm else None
    json.dump({'rmax_km': rmax, 'rne_km': rne, 'rne_over_rmax': ratio,
               'pk_threshold': 0.5},
              open(os.path.join(a.out, 'wez_rmax_rne.json'), 'w'), indent=1)
    print(f"\nDONE in {time.time()-t0:.0f}s -> {a.out}/wez_pk.json, wez_rmax_rne.json")

if __name__ == '__main__':
    main()
