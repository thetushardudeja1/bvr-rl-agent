"""
gen_wez_tacview.py — draw the Weapon-Employment-Zone *inside* a Tacview replay.

Tacview's text ACMI has no polygon primitive, so the WEZ is drawn as a ring of
small marker objects forming the petal boundary, anchored to the shooter and
re-positioned every frame -> the zone moves and rotates with the aircraft, like
a launch zone projected onto the world. Two petals are drawn around RED:
    R_max  (outer, CYAN)   -- kinematic reach vs a non-maneuvering target
    R_ne   (inner, ORANGE) -- no-escape range vs a best-evading target
so you can watch blue stay outside R_ne / beam its way out.

Petal geometry: radius as a function of angle off the shooter's nose.
    0deg  off nose (target ahead)  -> head-on aspect (180) -> largest range
    90deg off nose (target abeam)  -> beam aspect   (90)
    180deg off nose (target behind)-> tail aspect   (0)   -> smallest range

Data source: experiments/wez/wez_rmax_rne.json (fine sweep) if present, else
the coarse envelope.json + rne.json.

Run:  python gen_wez_tacview.py [seed] [--alt 8] [--vel 330]
"""
import sys, os, json, math, shutil, argparse
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.loggers import TacviewLogger

ap = argparse.ArgumentParser()
ap.add_argument('seed', nargs='?', type=int, default=7529)
ap.add_argument('--alt', type=int, default=8)     # km slice of the envelope
ap.add_argument('--vel', type=int, default=330)   # shooter-speed slice
ap.add_argument('--model', default='experiments/rir/rir_notch2_best')
ap.add_argument('--npts', type=int, default=48)   # boundary points per petal
A = ap.parse_args()

OUT = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/wez_tacview'
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- load WEZ data
def load_tables():
    """Return (rmax, rne) as dicts {aspect_deg: range_km} for the chosen slice."""
    fine = 'experiments/wez/wez_rmax_rne.json'
    if os.path.exists(fine):
        d = json.load(open(fine))
        if d.get('rmax_km'):
            rmax, rne = {}, {}
            for k, v in d['rmax_km'].items():
                # keys like asp90_alt8km_v330
                if f"_alt{A.alt}km_v{A.vel}" in k:
                    asp = int(k.split('_')[0][3:]); rmax[asp] = v
            for k, v in d['rne_km'].items():
                if f"_alt{A.alt}km_v{A.vel}" in k:
                    asp = int(k.split('_')[0][3:]); rne[asp] = v
            if rmax:
                print(f"using FINE sweep ({len(rmax)} aspects)")
                return rmax, rne
    # coarse fallback
    em = json.load(open('experiments/wez/envelope.json'))['rmax_km']
    rn = json.load(open('experiments/wez/rne.json'))['rne_km']
    rmax = {int(k.split('_')[0][3:]): v for k, v in em.items()
            if f"_alt{A.alt}km_v{A.vel}" in k}
    rne  = {int(k.split('_')[0][3:]): v for k, v in rn.items()
            if f"_alt{A.alt}km_v{A.vel}" in k}
    print(f"using COARSE envelope ({sorted(rmax)} aspects)")
    return rmax, rne

RMAX, RNE = load_tables()

def interp(table, aspect):
    xs = sorted(table)
    if aspect <= xs[0]:  return table[xs[0]]
    if aspect >= xs[-1]: return table[xs[-1]]
    for i in range(1, len(xs)):
        if aspect <= xs[i]:
            x0, x1 = xs[i-1], xs[i]
            t = (aspect - x0) / (x1 - x0)
            return table[x0] + t * (table[x1] - table[x0])
    return table[xs[-1]]

def radius_km(table, off_nose_deg):
    theta = off_nose_deg % 360.0
    if theta > 180.0: theta = 360.0 - theta      # symmetry L/R
    aspect = 180.0 - theta                         # nose=head-on, tail=away
    return interp(table, aspect)

# ---------------------------------------------------------------- geodesy
def offset(lat, lon, brg_deg, dist_m):
    b = math.radians(brg_deg)
    dlat = (dist_m * math.cos(b)) / 111_320.0
    dlon = (dist_m * math.sin(b)) / (111_320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon

# ---------------------------------------------------------------- run + draw
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(A.model, device='cpu')
env = BVRBase(baseEnv_conf); obs, _ = env.reset(seed=A.seed)
env.tacview_logger = TacviewLogger(env)
lg = env.tacview_logger

ID_RMAX = 0xE0000
ID_RNE  = 0xF0000

def draw_petals():
    r = env.red_agent
    try:
        lat = r.simObj.get_lat_gc_deg(); lon = r.simObj.get_long_gc_deg()
        alt = r.simObj.get_altitude();   psi = r.simObj.get_psi()
    except Exception:
        return
    for i in range(A.npts):
        a = 360.0 * i / A.npts
        brg = psi + a                                   # world bearing
        for table, base, color in ((RMAX, ID_RMAX, 'Cyan'), (RNE, ID_RNE, 'Orange')):
            rk = radius_km(table, a)
            if rk <= 0:
                continue
            mlat, mlon = offset(lat, lon, brg, rk * 1e3)
            oid = f"{base + i:X}"
            lg.lines.append(
                f"{oid},T={mlon:.7f}|{mlat:.7f}|{alt:.1f},"
                f"Type=Navaid+Static+Waypoint,Color={color}")

done = trunc = False
while not (done or trunc):
    lg.log_flight_data()
    draw_petals()
    a, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, info = env.step(a)
for _ in range(8):
    lg.log_flight_data(); draw_petals()
    a, _ = model.predict(obs, deterministic=True)
    try: obs, _, _, _, info = env.step(a)
    except Exception: break
lg.save_logs()

name = f"wez_seed{A.seed}_alt{A.alt}km_v{A.vel}_{info.get('outcome','?')}.acmi"
shutil.copy(os.path.join(baseEnv_conf.tacview_output_dir, 'BVRGym_engagement.acmi'),
            os.path.join(OUT, name))
print(f"SAVED: {os.path.join(OUT, name)}")
print(f"  Rmax aspects {RMAX}")
print(f"  Rne  aspects {RNE}")
