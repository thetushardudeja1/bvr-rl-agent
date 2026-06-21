"""
autopilot_benchmark_v3.py
==========================
Benchmark: ORIGINAL vs V3 "minimal fix"

V3 philosophy: Trust JSBSim's built-in F-16 FCS (which already has yaw damper,
roll-rate PID, g-load PID, alpha limiter). Only patch what JSBSim DOESN'T handle:

  1. Roll_max 80→45  (JSBSim limits roll RATE, not roll ANGLE — BVRGym sets the
     bank angle target, so capping it here prevents extreme 80° snap rolls)
  2. Integrator resets on large target jumps (pure BVRGym PID level — doesn't
     conflict with JSBSim's FCS at all)

NO yaw damper (JSBSim has one), NO pitch D-term change (JSBSim's g-load PID
handles pitch damping).
"""

import os, sys, math, json
import numpy as np

ROOT = os.path.expanduser("~/bvrgym_project_new")
sys.path.insert(0, ROOT)

from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.navigation import delta_heading


def apply_v3_patches(env):
    """V3: minimal fix — only Roll_max cap + integrator resets."""
    ac = env.blue_agent.simObj
    ap = ac.aircraft_control

    # FIX 1: Roll_max 80→45 (the ONLY thing JSBSim's FCS doesn't limit)
    ac.conf.aircraft_navigation.Roll_max = 45

    # FIX 2: Integrator resets on large target jumps (BVRGym PID level only)
    ac._prev_hdg = None
    ac._prev_alt = None
    _orig_step = ac._step

    def _reset_step(action):
        h, a = action[0], action[1]
        if ac._prev_hdg is not None:
            dh = abs(delta_heading(h, ac._prev_hdg))
            if dh > 15:
                ap.roll_PID.Integrator = 0.0
                ap.roll_PID.Derivator = 0.0
                ap.roll_sec_PID.Integrator = 0.0
        if ac._prev_alt is not None and abs(a - ac._prev_alt) > 500:
            ap.pitch_PID.Integrator = 0.0
            ap.pitch_PID.Derivator = 0.0
        ac._prev_hdg = h
        ac._prev_alt = a
        _orig_step(action)

    ac._step = _reset_step


def apply_v2_patches(env):
    """V2: full overhaul (yaw damper + pitch D=0 + roll cap 35)."""
    ac = env.blue_agent.simObj
    ap = ac.aircraft_control
    fdm = ac.fdm

    ac.conf.aircraft_navigation.Roll_max = 35
    ap.pitch_PID.Kd = 0.0
    ap.pitch_PID.Derivator = 0.0

    ac._yaw_rate_prev = 0.0
    ac._yaw_integral  = 0.0
    _orig_cmd = ac.command_aircraft

    def _yaw_damped_cmd(ail, elev, rud, thr):
        try:
            yr = fdm['velocities/r-rad_sec']
            rud = -1.5 * yr - 0.3 * (yr - ac._yaw_rate_prev)
            ac._yaw_integral += yr * 0.0833
            ac._yaw_integral = max(-0.5, min(0.5, ac._yaw_integral))
            rud -= 0.2 * ac._yaw_integral
            rud = max(-1.0, min(1.0, rud))
            ac._yaw_rate_prev = yr
        except: pass
        try:
            phi = fdm['attitude/phi-rad']
            lim = math.radians(35)
            if abs(phi) > lim:
                corr = -0.5 * (phi - math.copysign(lim, phi))
                ail = max(-1, min(1, ail + corr))
        except: pass
        _orig_cmd(ail, elev, rud, thr)

    ac.command_aircraft = _yaw_damped_cmd

    ac._prev_hdg = None
    ac._prev_alt = None
    _orig_step = ac._step

    def _reset_step(action):
        h, a = action[0], action[1]
        if ac._prev_hdg is not None:
            dh = abs(delta_heading(h, ac._prev_hdg))
            if dh > 15:
                ap.roll_PID.Integrator = 0.0
                ap.roll_PID.Derivator = 0.0
                ap.roll_sec_PID.Integrator = 0.0
        if ac._prev_alt is not None and abs(a - ac._prev_alt) > 500:
            ap.pitch_PID.Integrator = 0.0
            ap.pitch_PID.Derivator = 0.0
        ac._prev_hdg = h
        ac._prev_alt = a
        _orig_step(action)

    ac._step = _reset_step


# ─── HELPERS ──────────────────────────────────────────────────────

def hdg_to_action(h): return h / 180.0 - 1.0
def alt_to_action(a): return (a - 3000.0) / 4500.0 - 1.0

def get_init(env):
    ac = env.blue_agent.simObj
    return ac.get_psi(), ac.get_altitude()

def collect(env, n_steps, action_fn, label, patch_fn=None):
    obs, _ = env.reset()
    if patch_fn:
        patch_fn(env)
    ih, ia = get_init(env)
    fdm = env.blue_agent.simObj.fdm
    telem = []
    for i in range(n_steps):
        action = action_fn(i, obs, ih, ia)
        obs, rew, done, trunc, info = env.step(action)
        try:
            row = {
                'step': i,
                'roll_deg':    math.degrees(fdm['attitude/phi-rad']),
                'pitch_deg':   math.degrees(fdm['attitude/theta-rad']),
                'heading_deg': math.degrees(fdm['attitude/psi-rad']),
                'alt_ft':      fdm['position/h-sl-ft'],
                'roll_rate':   math.degrees(fdm['velocities/p-rad_sec']),
                'pitch_rate':  math.degrees(fdm['velocities/q-rad_sec']),
                'yaw_rate':    math.degrees(fdm['velocities/r-rad_sec']),
                'airspeed_kt': fdm['velocities/vc-kts'],
                'beta_deg':    math.degrees(fdm['aero/beta-rad']),
                'g_load':      fdm['accelerations/Nz'],
            }
        except Exception as e:
            row = {'step': i, 'error': str(e)}
        telem.append(row)
        if done or trunc: break
    return telem

def analyze(telem):
    n = len(telem)
    if n < 2: return {}
    def col(k): return [t.get(k, 0) for t in telem]
    rolls = col('roll_deg'); pitches = col('pitch_deg')
    yaw_rates = col('yaw_rate'); betas = col('beta_deg')
    g_loads = col('g_load'); speeds = col('airspeed_kt')
    alts = col('alt_ft')
    d_roll = [abs(rolls[i]-rolls[i-1]) for i in range(1,n)]
    d_pitch = [abs(pitches[i]-pitches[i-1]) for i in range(1,n)]
    roll_rev = sum(1 for i in range(2,n) if (rolls[i]-rolls[i-1])*(rolls[i-1]-rolls[i-2])<0)
    pitch_rev = sum(1 for i in range(2,n) if (pitches[i]-pitches[i-1])*(pitches[i-1]-pitches[i-2])<0)
    return {
        'roll_std': round(np.std(rolls),2), 'roll_max': round(max(abs(r) for r in rolls),2),
        'roll_d': round(np.mean(d_roll),3), 'roll_rev': roll_rev,
        'pitch_std': round(np.std(pitches),2), 'pitch_d': round(np.mean(d_pitch),3),
        'pitch_rev': pitch_rev,
        'yaw_max': round(max(abs(y) for y in yaw_rates),3),
        'beta_max': round(max(abs(b) for b in betas),3),
        'alt_std': round(np.std(alts),1), 'g_max': round(max(abs(g) for g in g_loads),3),
        'spd_min': round(min(speeds),1), 'spd_mean': round(np.mean(speeds),1),
    }


def print_3way(label, orig, v2, v3):
    print(f"\n{'─'*75}")
    print(f"  {label}")
    print(f"{'─'*75}")
    print(f"  {'Metric':<22} {'ORIGINAL':>10} {'V2':>10} {'V3':>10} {'Best':>8}")
    print(f"  {'─'*62}")
    metrics = [
        ('Roll σ°',       'roll_std',  'lo'),
        ('Roll max°',     'roll_max',  'lo'),
        ('Roll Δ/step°',  'roll_d',    'lo'),
        ('Roll reversals','roll_rev',  'lo'),
        ('Pitch σ°',      'pitch_std', 'lo'),
        ('Pitch Δ/step°', 'pitch_d',   'lo'),
        ('Pitch rev',     'pitch_rev', 'lo'),
        ('Yaw rate max',  'yaw_max',   'lo'),
        ('β max°',        'beta_max',  'lo'),
        ('Alt σ ft',      'alt_std',   'lo'),
        ('G max',         'g_max',     'lo'),
        ('Speed min kt',  'spd_min',   'hi'),
        ('Speed mean kt', 'spd_mean',  'hi'),
    ]
    scores = {'ORIG': 0, 'V2': 0, 'V3': 0}
    for name, key, direction in metrics:
        ov = orig.get(key, 0); v2v = v2.get(key, 0); v3v = v3.get(key, 0)
        vals = {'ORIG': ov, 'V2': v2v, 'V3': v3v}
        if direction == 'lo':
            best = min(vals, key=vals.get)
        else:
            best = max(vals, key=vals.get)
        scores[best] += 1
        tag = f"{'✅'+best:>8}"
        print(f"  {name:<22} {ov:>10} {v2v:>10} {v3v:>10} {tag}")
    return scores


# ─── ACTIONS ──────────────────────────────────────────────────────

def straight(s, o, ih, ia):
    return np.array([hdg_to_action(ih), alt_to_action(ia), 0.5])

def right_turn(s, o, ih, ia):
    return np.array([hdg_to_action((ih+90)%360), alt_to_action(ia), 0.5])

def left_turn(s, o, ih, ia):
    return np.array([hdg_to_action((ih-90)%360), alt_to_action(ia), 0.5])

def climb_desc(s, o, ih, ia):
    if s < 100:
        return np.array([hdg_to_action(ih), alt_to_action(ia+1500), 0.5])
    return np.array([hdg_to_action(ih), alt_to_action(ia-1500), 0.5])

_rng_state = {}
def rapid(s, o, ih, ia):
    if 'rng' not in _rng_state:
        _rng_state['rng'] = np.random.RandomState(42)
    a = _rng_state['rng'].uniform(-1, 1, size=3)
    a[2] = abs(a[2])
    return a

def reset_rapid_rng():
    _rng_state.clear()


# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("  3-WAY BENCHMARK: ORIGINAL vs V2 vs V3")
    print("  V3 = JSBSim FCS + Roll_max=45 + Integrator Resets (minimal fix)")
    print("=" * 75)

    baseEnv_conf.tacview_output = False
    outdir = os.path.join(ROOT, "experiments", "v2")
    os.makedirs(outdir, exist_ok=True)

    tests = [
        ("Straight & Level",  straight,   200),
        ("90° Right Turn",    right_turn, 150),
        ("90° Left Turn",     left_turn,  150),
        ("Climb & Descend",   climb_desc, 200),
        ("Rapid Random",      rapid,      200),
    ]

    total_scores = {'ORIG': 0, 'V2': 0, 'V3': 0}

    for test_name, action_fn, n_steps in tests:
        print(f"\n{'━'*75}")
        print(f"  {test_name} ({n_steps} steps)")
        print(f"{'━'*75}")

        # ORIGINAL
        reset_rapid_rng()
        env = BVRBase(baseEnv_conf)
        t_orig = collect(env, n_steps, action_fn, "ORIG", patch_fn=None)
        env.close()
        r_orig = analyze(t_orig)

        # V2
        reset_rapid_rng()
        env = BVRBase(baseEnv_conf)
        t_v2 = collect(env, n_steps, action_fn, "V2", patch_fn=apply_v2_patches)
        env.close()
        r_v2 = analyze(t_v2)

        # V3
        reset_rapid_rng()
        env = BVRBase(baseEnv_conf)
        t_v3 = collect(env, n_steps, action_fn, "V3", patch_fn=apply_v3_patches)
        env.close()
        r_v3 = analyze(t_v3)

        scores = print_3way(test_name, r_orig, r_v2, r_v3)
        for k in scores:
            total_scores[k] += scores[k]

    # ── FINAL ──
    print(f"\n{'='*75}")
    print(f"  FINAL SCORECARD (total metric wins across all tests)")
    print(f"{'='*75}")
    for k, v in sorted(total_scores.items(), key=lambda x: -x[1]):
        bar = '█' * v
        print(f"  {k:<8} {v:>3} wins  {bar}")

    best = max(total_scores, key=total_scores.get)
    print(f"\n  🏆 WINNER: {best}")
    if best == 'V3':
        print("  V3 (JSBSim FCS + Roll_max=45 + Integrator Resets) is the best!")
        print("  → Trust JSBSim's built-in fly-by-wire, only cap roll angle.")
    elif best == 'ORIG':
        print("  Original autopilot is hard to beat — JSBSim's FCS is good!")
        print("  → Consider just reducing Roll_max without other changes.")
    else:
        print("  V2 won despite double-damping — might need further investigation.")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
