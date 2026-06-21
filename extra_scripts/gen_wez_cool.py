"""
gen_wez_cool.py — a *cool* Weapon-Employment-Zone overlay inside Tacview.

Draws, anchored to RED and rotating with its nose every frame:
  * a FILLED Pk heatmap petal  -- dense dots coloured by kill probability:
        Red    (Pk >= 0.66)  lethal core
        Orange (0.33-0.66)
        Yellow (< 0.33)      fringe
  * a bright CYAN R_max boundary ring (kinematic reach)
  * an animated WHITE radar-sweep spoke rotating like a PPI scope
  * floating range labels at the nose (R_max / R_ne)

Pk source: experiments/wez/wez_pk.json (full Monte-Carlo grid) if it has fine
aspect resolution; otherwise a smooth Pk field synthesised from R_max / R_ne.

Run:  python gen_wez_cool.py [seed] [--alt 8] [--vel 330]
"""
import sys, os, json, math, shutil, argparse
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.loggers import TacviewLogger

ap = argparse.ArgumentParser()
ap.add_argument('seed', nargs='?', type=int, default=7529)
ap.add_argument('--alt', type=int, default=8)
ap.add_argument('--vel', type=int, default=330)
ap.add_argument('--model', default='experiments/rir/rir_notch2_best')
ap.add_argument('--rings', type=int, default=9)     # range shells for the fill
ap.add_argument('--spokes', type=int, default=24)   # angular resolution of fill
A = ap.parse_args()

OUT = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/wez_tacview'
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------- load WEZ tables
def slice_keys(d):
    out = {}
    for k, v in d.items():
        if f"_alt{A.alt}km_v{A.vel}" in k:
            out[int(k.split('_')[0][3:])] = v
    return out

PKGRID = None
def load():
    global PKGRID
    fine = 'experiments/wez/wez_rmax_rne.json'
    pkf  = 'experiments/wez/wez_pk.json'
    if os.path.exists(pkf):
        pk = json.load(open(pkf))
        asp = pk.get('aspects', [])
        if len(asp) > 4:                                  # fine grid available
            PKGRID = pk
    if os.path.exists(fine):
        d = json.load(open(fine))
        rmax, rne = slice_keys(d.get('rmax_km', {})), slice_keys(d.get('rne_km', {}))
        if rmax:
            print(f"FINE rmax/rne ({sorted(rmax)})")
            return rmax, rne
    em = json.load(open('experiments/wez/envelope.json'))['rmax_km']
    rn = json.load(open('experiments/wez/rne.json'))['rne_km']
    print("COARSE envelope")
    return slice_keys(em), slice_keys(rn)

RMAX, RNE = load()

def interp(table, x):
    xs = sorted(table)
    if not xs: return 0
    if x <= xs[0]: return table[xs[0]]
    if x >= xs[-1]: return table[xs[-1]]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            a, b = xs[i-1], xs[i]
            t = (x - a) / (b - a)
            return table[a] + t * (table[b] - table[a])
    return table[xs[-1]]

def fold_aspect(off_nose):
    th = off_nose % 360.0
    if th > 180: th = 360 - th
    return 180.0 - th                                      # nose=head-on

def rmax_km(off_nose): return interp(RMAX, fold_aspect(off_nose))
def rne_km(off_nose):  return interp(RNE,  fold_aspect(off_nose))

def pk_at(off_nose, d_km):
    """Kill probability at a point, from the fine grid if present else a model."""
    asp = fold_aspect(off_nose)
    if PKGRID is not None:
        prefix = f"asp{min(PKGRID['aspects'], key=lambda x: abs(x-asp))}_alt{A.alt}km_v{A.vel}_evade"
        cell = PKGRID['pk'].get(prefix, {})
        if cell:
            ranges = sorted(int(r) for r in cell)
            rr = min(ranges, key=lambda x: abs(x - d_km))
            return cell.get(str(rr), 0.0)
    Rn, Rm = rne_km(off_nose), rmax_km(off_nose)
    if d_km <= Rn:  return 1.0
    if d_km <= Rm and Rm > Rn: return 1.0 - (d_km - Rn) / (Rm - Rn)
    return 0.0

def pk_color(pk):
    return 'Red' if pk >= 0.66 else 'Orange' if pk >= 0.33 else 'Yellow'

# ----------------------------------------------------------- geodesy
def offset(lat, lon, brg, dist_m):
    b = math.radians(brg)
    return (lat + (dist_m * math.cos(b)) / 111_320.0,
            lon + (dist_m * math.sin(b)) / (111_320.0 * math.cos(math.radians(lat))))

# ----------------------------------------------------------- run + draw
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(A.model, device='cpu')
env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=A.seed)
env.tacview_logger = TacviewLogger(env); lg = env.tacview_logger

B_FILL, B_RING, B_SWEEP, B_LABEL = 0xA00000, 0xB00000, 0xC00000, 0xD00000
frame = 0

def draw(frame):
    r = env.red_agent
    try:
        lat = r.simObj.get_lat_gc_deg(); lon = r.simObj.get_long_gc_deg()
        alt = r.simObj.get_altitude();   psi = r.simObj.get_psi()
    except Exception:
        return
    oid = 0
    # filled Pk heatmap
    for s in range(A.spokes):
        a = 360.0 * s / A.spokes
        brg = psi + a
        Rm = rmax_km(a)
        if Rm <= 0: continue
        for j in range(1, A.rings + 1):
            d = Rm * j / A.rings
            pk = pk_at(a, d)
            if pk <= 0.02: continue
            mlat, mlon = offset(lat, lon, brg, d * 1e3)
            lg.lines.append(f"{B_FILL+oid:X},T={mlon:.6f}|{mlat:.6f}|{alt:.0f},"
                            f"Type=Misc+Shrapnel,Color={pk_color(pk)}")
            oid += 1
    # bright R_max boundary ring
    for s in range(72):
        a = 360.0 * s / 72
        Rm = rmax_km(a)
        if Rm <= 0: continue
        mlat, mlon = offset(lat, lon, psi + a, Rm * 1e3)
        lg.lines.append(f"{B_RING+s:X},T={mlon:.6f}|{mlat:.6f}|{alt:.0f},"
                        f"Type=Navaid+Static+Waypoint,Color=Cyan")
    # animated radar sweep spoke
    sweep_brg = psi + (frame * 13) % 360
    Rm = rmax_km((frame * 13) % 360)
    for j in range(12):
        d = max(Rm, 5) * (j + 1) / 12
        mlat, mlon = offset(lat, lon, sweep_brg, d * 1e3)
        lg.lines.append(f"{B_SWEEP+j:X},T={mlon:.6f}|{mlat:.6f}|{alt:.0f},"
                        f"Type=Misc+Shrapnel,Color=White")
    # floating labels at the nose
    rm0, rn0 = rmax_km(0), rne_km(0)
    for li, (rk, txt, col) in enumerate([(rm0, f"R_max {rm0:.0f}km", 'Cyan'),
                                         (rn0, f"R_NE {rn0:.0f}km", 'Red')]):
        if rk <= 0: continue
        mlat, mlon = offset(lat, lon, psi, rk * 1e3)
        lg.lines.append(f"{B_LABEL+li:X},T={mlon:.6f}|{mlat:.6f}|{alt+300:.0f},"
                        f"Type=Navaid+Static+Waypoint,Color={col},Name={txt}")

done = trunc = False
while not (done or trunc):
    lg.log_flight_data(); draw(frame); frame += 1
    act, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, info = env.step(act)
for _ in range(8):
    lg.log_flight_data(); draw(frame); frame += 1
    act, _ = model.predict(obs, deterministic=True)
    try: obs, _, _, _, info = env.step(act)
    except Exception: break
lg.save_logs()

name = f"wezCOOL_seed{A.seed}_alt{A.alt}km_v{A.vel}_{info.get('outcome','?')}.acmi"
shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
            os.path.join(OUT, name))
print(f"SAVED: {os.path.join(OUT, name)}")
print(f"  Pk source: {'FINE grid' if PKGRID else 'Rmax/Rne model'}")
print(f"  Rmax {RMAX}\n  Rne  {RNE}")
