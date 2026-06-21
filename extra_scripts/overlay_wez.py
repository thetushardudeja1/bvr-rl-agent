"""
overlay_wez.py — post-process an ACMI to inject the BLUE shooter's measured
weapon-engagement zone as a moving Dynamic-Launch-Zone (DLZ) lobe:
  outer ring (yellow) = Rmax,   inner ring (orange) = Rne (no-escape).
The lobe is computed each frame from blue's altitude/speed and the target's
heading (so it bulges toward a head-on target, pinches toward the tail), and
moves with the shooter. Usage: python overlay_wez.py in.acmi out.acmi
"""
import sys, re, json, math, os

IN, OUT = sys.argv[1], sys.argv[2]
NB = 16                       # bearings per ring
ASPECTS, ALTS, VELS = (0, 90, 180), (5, 8, 11), (250, 330, 400)

def load(p, key):
    try: return json.load(open(p))[key]
    except Exception: return {}
rmax = load(os.path.expanduser('~/bvr_analysis/experiments/wez/envelope.json'), 'rmax_km')
rne  = load(os.path.expanduser('~/bvr_analysis/experiments/wez/rne.json'), 'rne_km')

def nearest(v, opts): return min(opts, key=lambda o: abs(o - v))
def dh(a, b): return (a - b + 180) % 360 - 180

def lookup(table, asp, alt, vel):
    a, h, s = nearest(asp, ASPECTS), nearest(alt, ALTS), nearest(vel, VELS)
    return table.get(f"asp{a}_alt{h}km_v{s}", table.get(f"asp{a}_alt{h}km", 0))

# parse blue (1000) and red (2000) per-frame state
frames = []   # list of (t, blue(lon,lat,alt,yaw), red_yaw)
t = 0.0; b = None; r_yaw = None
lines = open(IN, errors='ignore').read().splitlines()
for ln in lines:
    if ln.startswith('#'):
        if b: frames.append((t, b, r_yaw))
        try: t = float(ln[1:])
        except: pass
        b = None
        continue
    m = re.match(r'^(1000|2000),T=([^,]+)', ln)
    if not m: continue
    p = m.group(2).split('|')
    if len(p) < 6: continue
    try: lon, lat, alt, yaw = float(p[0]), float(p[1]), float(p[2]), float(p[5])
    except: continue
    if m.group(1) == '1000': b = (lon, lat, alt, yaw)
    else: r_yaw = yaw
if b: frames.append((t, b, r_yaw))

def speed_kmps(i):
    if i == 0 or i >= len(frames): return 0.33
    (t0, b0, _), (t1, b1, _) = frames[i-1], frames[i]
    dt = max(0.1, t1 - t0)
    dx = (b1[0]-b0[0])*111.32*math.cos(math.radians(b0[1])); dy = (b1[1]-b0[1])*111.32
    return math.hypot(dx, dy) / dt

# rebuild ACMI with lobe objects (IDs 5000.. Rmax, 5200.. Rne) updated each frame
out = []
fi = -1
def lobe_lines(b, r_yaw, vkmps):
    lon, lat, alt, _ = b
    vms = vkmps * 1000.0
    res = []
    for j in range(NB):
        th = 360.0 * j / NB
        bearing_back = (th + 180) % 360
        asp = 180 - abs(dh(bearing_back, r_yaw if r_yaw is not None else 0.0))
        for table, base, in ((rmax, 5000), (rne, 5200)):
            R = lookup(table, asp, alt/1e3, vms)
            dlat = R*math.cos(math.radians(th))/111.32
            dlon = R*math.sin(math.radians(th))/(111.32*math.cos(math.radians(lat)))
            res.append(f"{base+j},T={lon+dlon:.6f}|{lat+dlat:.6f}|{alt:.1f}")
    return res

declared = False
for ln in lines:
    out.append(ln)
    if ln.startswith('#'):
        fi += 1
        if 0 <= fi < len(frames):
            _, bb, ry = frames[fi]
            if not declared:   # one-time object declarations (color/type)
                for j in range(NB):
                    out.append(f"{5000+j},Type=Navaid+Static+Waypoint,Color=Yellow,Name=WEZ_Rmax")
                    out.append(f"{5200+j},Type=Navaid+Static+Waypoint,Color=Orange,Name=WEZ_Rne")
                declared = True
            out += lobe_lines(bb, ry, speed_kmps(fi))
open(OUT, 'w').write('\n'.join(out) + '\n')
print(f"WEZ overlay written -> {OUT}  ({len(frames)} frames, {NB} bearings x 2 rings)")
