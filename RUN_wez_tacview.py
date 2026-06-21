"""
================================================================================
 RUN_wez_tacview.py  --  PRESS ▶ "RUN" TO MAKE THE COOL WEZ OVERLAY REPLAY
================================================================================
No terminal needed. Open in VS Code, click Run (▶) or press F5.

Generates a Tacview replay with red's Weapon-Employment-Zone drawn IN the world,
anchored to red and rotating with its nose:
  * filled Pk heatmap petal  (Red core -> Orange -> Yellow fringe)
  * cyan R_max boundary ring
  * an animated white radar-sweep spoke
  * floating R_max / R_NE labels

Uses the fine Monte-Carlo grid (experiments/wez/wez_*.json) if present.
Output -> ./output_tacview/
================================================================================
"""
import os, sys, json, math, shutil
try:                                    # Windows cp1252 can't print emoji/arrows
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)

# ============================== CONFIG ========================================
SEED  = 7529          # which engagement to replay
ALT   = 8             # km slice of the envelope to draw
VEL   = 330           # m/s shooter-speed slice
MODEL = 'experiments/rir/rir_notch2_best'
SPOKES, RINGS = 24, 9         # fill resolution
# ==============================================================================

OUT = os.path.join(HERE, 'output_tacview'); os.makedirs(OUT, exist_ok=True)
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.loggers import TacviewLogger

def slice_keys(d):
    out = {}
    for k, v in d.items():
        if f"_alt{ALT}km_v{VEL}" in k:
            out[int(k.split('_')[0][3:])] = v
    return out

PKGRID = None
def load():
    global PKGRID
    pkf = 'experiments/wez/wez_pk.json'; fine = 'experiments/wez/wez_rmax_rne.json'
    if os.path.exists(pkf):
        pk = json.load(open(pkf))
        if len(pk.get('aspects', [])) > 4:
            PKGRID = pk
    if os.path.exists(fine):
        d = json.load(open(fine))
        rmax, rne = slice_keys(d.get('rmax_km', {})), slice_keys(d.get('rne_km', {}))
        if rmax:
            print("using FINE sweep"); return rmax, rne
    em = json.load(open('experiments/wez/envelope.json'))['rmax_km']
    rn = json.load(open('experiments/wez/rne.json'))['rne_km']
    print("using COARSE envelope"); return slice_keys(em), slice_keys(rn)

RMAX, RNE = load()

def interp(t, x):
    xs = sorted(t)
    if not xs: return 0
    if x <= xs[0]: return t[xs[0]]
    if x >= xs[-1]: return t[xs[-1]]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            a, b = xs[i-1], xs[i]; f = (x-a)/(b-a)
            return t[a] + f*(t[b]-t[a])
    return t[xs[-1]]

def fold(off):
    th = off % 360.0
    if th > 180: th = 360-th
    return 180.0 - th

def rmax_km(o): return interp(RMAX, fold(o))
def rne_km(o):  return interp(RNE,  fold(o))

def pk_at(o, d):
    asp = fold(o)
    if PKGRID is not None:
        key = f"asp{min(PKGRID['aspects'], key=lambda x: abs(x-asp))}_alt{ALT}km_v{VEL}_evade"
        cell = PKGRID['pk'].get(key, {})
        if cell:
            rr = min((int(r) for r in cell), key=lambda x: abs(x-d))
            return cell.get(str(rr), 0.0)
    Rn, Rm = rne_km(o), rmax_km(o)
    if d <= Rn: return 1.0
    if d <= Rm and Rm > Rn: return 1.0 - (d-Rn)/(Rm-Rn)
    return 0.0

def col(pk): return 'Red' if pk >= 0.66 else 'Orange' if pk >= 0.33 else 'Yellow'

def offset(lat, lon, brg, m):
    b = math.radians(brg)
    return (lat + (m*math.cos(b))/111_320.0,
            lon + (m*math.sin(b))/(111_320.0*math.cos(math.radians(lat))))

baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')
env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=SEED)
env.tacview_logger = TacviewLogger(env); lg = env.tacview_logger
B_FILL, B_RING, B_SWEEP, B_LABEL = 0xA00000, 0xB00000, 0xC00000, 0xD00000

def draw(frame):
    r = env.red_agent
    try:
        lat = r.simObj.get_lat_gc_deg(); lon = r.simObj.get_long_gc_deg()
        alt = r.simObj.get_altitude();   psi = r.simObj.get_psi()
    except Exception:
        return
    oid = 0
    for s in range(SPOKES):
        a = 360.0*s/SPOKES; Rm = rmax_km(a)
        if Rm <= 0: continue
        for j in range(1, RINGS+1):
            d = Rm*j/RINGS; pk = pk_at(a, d)
            if pk <= 0.02: continue
            mlat, mlon = offset(lat, lon, psi+a, d*1e3)
            lg.lines.append(f"{B_FILL+oid:X},T={mlon:.6f}|{mlat:.6f}|{alt:.0f},"
                            f"Type=Misc+Shrapnel,Color={col(pk)}"); oid += 1
    for s in range(72):
        a = 360.0*s/72; Rm = rmax_km(a)
        if Rm <= 0: continue
        mlat, mlon = offset(lat, lon, psi+a, Rm*1e3)
        lg.lines.append(f"{B_RING+s:X},T={mlon:.6f}|{mlat:.6f}|{alt:.0f},"
                        f"Type=Navaid+Static+Waypoint,Color=Cyan")
    sb = psi + (frame*13) % 360; Rm = rmax_km((frame*13) % 360)
    for j in range(12):
        d = max(Rm, 5)*(j+1)/12
        mlat, mlon = offset(lat, lon, sb, d*1e3)
        lg.lines.append(f"{B_SWEEP+j:X},T={mlon:.6f}|{mlat:.6f}|{alt:.0f},"
                        f"Type=Misc+Shrapnel,Color=White")
    for li, (rk, txt, c) in enumerate([(rmax_km(0), f"R_max {rmax_km(0):.0f}km", 'Cyan'),
                                       (rne_km(0), f"R_NE {rne_km(0):.0f}km", 'Red')]):
        if rk <= 0: continue
        mlat, mlon = offset(lat, lon, psi, rk*1e3)
        lg.lines.append(f"{B_LABEL+li:X},T={mlon:.6f}|{mlat:.6f}|{alt+300:.0f},"
                        f"Type=Navaid+Static+Waypoint,Color={c},Name={txt}")

print("Running engagement and drawing WEZ ...")
done = trunc = False; frame = 0
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
name = f"WEZ_seed{SEED}_alt{ALT}km_v{VEL}_{info.get('outcome','?')}.acmi"
shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
            os.path.join(OUT, name))
print(f"\nDONE -> {os.path.join(OUT, name)}")
print(f"  Pk source: {'FINE grid' if PKGRID else 'Rmax/Rne model'}")
