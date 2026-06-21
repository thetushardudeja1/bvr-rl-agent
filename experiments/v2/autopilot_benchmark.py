"""
autopilot_stability_test_v2.py
===============================
Exhaustive stability diagnostic that compares:
  A) ORIGINAL autopilot (no patches — the baseline that got 65%)
  B) V2 autopilot (yaw damper, pitch D=0, roll cap 35°)

Uses correct action mapping so "straight & level" actually holds
the aircraft's initial heading and altitude.

Action mapping (from BaseEnv.from_nn2agent):
  action[0] in [-1,1] → heading = (a+1)*180 + 0   → [0°, 360°]
  action[1] in [-1,1] → altitude = (a+1)*4500 + 3000  → [3000m, 12000m]
  action[2] ≤ 0 → throttle=0.49;  action[2] > 0 → throttle=0.69

So to hold heading H°:  action[0] = H/180 - 1
   to hold altitude Am: action[1] = (A - 3000)/4500 - 1
"""

import os, sys, math, json
import numpy as np

ROOT = os.path.expanduser("~/bvrgym_project_new")
sys.path.insert(0, ROOT)

from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.navigation import delta_heading


# ─── V2 PATCHES (applied per-instance after reset) ───────────────

def apply_v2_patches(env):
    """Apply V2 fixes to the blue aircraft's autopilot instance."""
    ac = env.blue_agent.simObj          # F16BVR
    ap = ac.aircraft_control            # AircraftPIDAutopilot
    fdm = ac.fdm                        # JSBSim FDM

    # FIX 1: Roll_max 80→35
    ac.conf.aircraft_navigation.Roll_max = 35

    # FIX 2: Pitch PID D-term 1.0→0.0 (stops porpoising)
    ap.pitch_PID.Kd = 0.0
    ap.pitch_PID.Derivator = 0.0

    # FIX 3: Yaw damper via command_aircraft wrapper
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
        # Roll hard-limit
        try:
            phi = fdm['attitude/phi-rad']
            lim = math.radians(35)
            if abs(phi) > lim:
                corr = -0.5 * (phi - math.copysign(lim, phi))
                ail = max(-1, min(1, ail + corr))
        except: pass
        _orig_cmd(ail, elev, rud, thr)

    ac.command_aircraft = _yaw_damped_cmd

    # FIX 4: Integrator resets on large target jumps
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

def hdg_to_action(hdg_deg):
    """Convert heading in [0,360] to action in [-1,1]."""
    return hdg_deg / 180.0 - 1.0

def alt_to_action(alt_m):
    """Convert altitude in meters [3000,12000] to action in [-1,1]."""
    return (alt_m - 3000.0) / 4500.0 - 1.0

def get_initial_state(env):
    """Read blue aircraft's initial heading (deg) and altitude (m)."""
    ac = env.blue_agent.simObj
    return ac.get_psi(), ac.get_altitude()


def collect(env, n_steps, action_fn, label, apply_patches=False):
    """Run n_steps and collect full FDM telemetry."""
    obs, _ = env.reset()
    if apply_patches:
        apply_v2_patches(env)

    init_hdg, init_alt = get_initial_state(env)
    fdm = env.blue_agent.simObj.fdm
    telem = []

    for i in range(n_steps):
        action = action_fn(i, obs, init_hdg, init_alt)
        obs, rew, done, trunc, info = env.step(action)
        try:
            row = {
                'step': i, 'label': label,
                'alt_ft':      fdm['position/h-sl-ft'],
                'roll_deg':    math.degrees(fdm['attitude/phi-rad']),
                'pitch_deg':   math.degrees(fdm['attitude/theta-rad']),
                'heading_deg': math.degrees(fdm['attitude/psi-rad']),
                'roll_rate':   math.degrees(fdm['velocities/p-rad_sec']),
                'pitch_rate':  math.degrees(fdm['velocities/q-rad_sec']),
                'yaw_rate':    math.degrees(fdm['velocities/r-rad_sec']),
                'airspeed_kt': fdm['velocities/vc-kts'],
                'mach':        fdm['velocities/mach'],
                'alpha_deg':   math.degrees(fdm['aero/alpha-rad']),
                'beta_deg':    math.degrees(fdm['aero/beta-rad']),
                'g_load':      fdm['accelerations/Nz'],
                'aileron':     fdm['fcs/aileron-cmd-norm'],
                'elevator':    fdm['fcs/elevator-cmd-norm'],
                'rudder':      fdm['fcs/rudder-cmd-norm'],
            }
        except Exception as e:
            row = {'step': i, 'label': label, 'error': str(e)}
        telem.append(row)
        if done or trunc:
            break
    return telem


def analyze(telem, label):
    n = len(telem)
    if n < 2:
        return {'label': label, 'steps': n}
    def col(k): return [t.get(k, 0) for t in telem]

    rolls = col('roll_deg'); pitches = col('pitch_deg')
    headings = col('heading_deg'); alts = col('alt_ft')
    yaw_rates = col('yaw_rate'); betas = col('beta_deg')
    g_loads = col('g_load'); speeds = col('airspeed_kt')

    d_roll = [abs(rolls[i]-rolls[i-1]) for i in range(1,n)]
    d_pitch = [abs(pitches[i]-pitches[i-1]) for i in range(1,n)]
    roll_rev = sum(1 for i in range(2,n) if (rolls[i]-rolls[i-1])*(rolls[i-1]-rolls[i-2])<0)
    pitch_rev = sum(1 for i in range(2,n) if (pitches[i]-pitches[i-1])*(pitches[i-1]-pitches[i-2])<0)

    return {
        'label': label, 'steps': n,
        'roll_std':       round(np.std(rolls), 2),
        'roll_max_abs':   round(max(abs(r) for r in rolls), 2),
        'roll_d_mean':    round(np.mean(d_roll), 3),
        'roll_reversals': roll_rev,
        'pitch_std':      round(np.std(pitches), 2),
        'pitch_d_mean':   round(np.mean(d_pitch), 3),
        'pitch_reversals':pitch_rev,
        'yaw_rate_max':   round(max(abs(y) for y in yaw_rates), 3),
        'beta_max':       round(max(abs(b) for b in betas), 3),
        'alt_std':        round(np.std(alts), 1),
        'alt_drift':      round(alts[-1]-alts[0], 1),
        'g_max':          round(max(abs(g) for g in g_loads), 3),
        'speed_min':      round(min(speeds), 1),
        'speed_mean':     round(np.mean(speeds), 1),
    }


def print_compare(label, orig, v2):
    """Print side-by-side comparison."""
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  {'Metric':<25} {'ORIGINAL':>12} {'V2':>12} {'Better?':>10}")
    print(f"  {'─'*59}")

    metrics = [
        ('Roll σ (°)',         'roll_std',       'lower'),
        ('Roll max (°)',       'roll_max_abs',   'lower'),
        ('Roll Δ/step (°)',    'roll_d_mean',    'lower'),
        ('Roll reversals',     'roll_reversals', 'lower'),
        ('Pitch σ (°)',        'pitch_std',      'lower'),
        ('Pitch Δ/step (°)',   'pitch_d_mean',   'lower'),
        ('Pitch reversals',    'pitch_reversals','lower'),
        ('Yaw rate max (°/s)', 'yaw_rate_max',   'lower'),
        ('Sideslip β max (°)', 'beta_max',       'lower'),
        ('Alt σ (ft)',         'alt_std',        'lower'),
        ('G-load max',         'g_max',          'lower'),
        ('Speed min (kt)',     'speed_min',      'higher'),
        ('Speed mean (kt)',    'speed_mean',     'higher'),
    ]

    for name, key, direction in metrics:
        ov = orig.get(key, 0)
        vv = v2.get(key, 0)
        if direction == 'lower':
            winner = 'V2 ✅' if vv < ov else ('ORIG' if vv > ov else 'TIE')
        else:
            winner = 'V2 ✅' if vv > ov else ('ORIG' if vv < ov else 'TIE')
        print(f"  {name:<25} {ov:>12} {vv:>12} {winner:>10}")


# ─── ACTION FUNCTIONS ─────────────────────────────────────────────

def straight_level(step, obs, init_hdg, init_alt):
    """Hold the initial heading and altitude."""
    return np.array([hdg_to_action(init_hdg), alt_to_action(init_alt), 0.5])

def turn_right(step, obs, init_hdg, init_alt):
    """Turn 90° right from initial heading."""
    target = (init_hdg + 90) % 360
    return np.array([hdg_to_action(target), alt_to_action(init_alt), 0.5])

def turn_left(step, obs, init_hdg, init_alt):
    """Turn 90° left from initial heading."""
    target = (init_hdg - 90) % 360
    return np.array([hdg_to_action(target), alt_to_action(init_alt), 0.5])

def climb_descend(step, obs, init_hdg, init_alt):
    """Climb for first half, descend for second half."""
    if step < 100:
        return np.array([hdg_to_action(init_hdg), alt_to_action(init_alt + 1500), 0.5])
    else:
        return np.array([hdg_to_action(init_hdg), alt_to_action(init_alt - 1500), 0.5])

def rapid_random(step, obs, init_hdg, init_alt, _rng=[None]):
    """RL-like random commands every step."""
    if _rng[0] is None:
        _rng[0] = np.random.RandomState(42)
    a = _rng[0].uniform(-1, 1, size=3)
    a[2] = abs(a[2])
    return a


# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  AUTOPILOT BENCHMARK: ORIGINAL vs V2")
    print("  Side-by-side stability comparison")
    print("=" * 70)

    baseEnv_conf.tacview_output = False
    outdir = os.path.join(ROOT, "experiments", "v2")
    os.makedirs(outdir, exist_ok=True)

    tests = [
        ("TEST 1: Straight & Level (200 steps)", straight_level, 200),
        ("TEST 2: 90° Right Turn (150 steps)",   turn_right,     150),
        ("TEST 3: 90° Left Turn (150 steps)",    turn_left,      150),
        ("TEST 4: Climb & Descend (200 steps)",  climb_descend,  200),
        ("TEST 5: Rapid Random (200 steps)",     rapid_random,   200),
    ]

    all_results = []
    all_telem = {}

    for test_name, action_fn, n_steps in tests:
        print(f"\n{'━'*70}")
        print(f"  {test_name}")
        print(f"{'━'*70}")

        # ── ORIGINAL (no patches) ──
        print("  Running ORIGINAL autopilot...")
        env = BVRBase(baseEnv_conf)
        telem_orig = collect(env, n_steps, action_fn, f"{test_name} [ORIG]", apply_patches=False)
        env.close()
        r_orig = analyze(telem_orig, f"{test_name} [ORIG]")

        # ── V2 (with patches) ──
        print("  Running V2 autopilot...")
        # Reset the random seed for rapid_random so both get same commands
        rapid_random.__defaults__ = (None, [None])
        env = BVRBase(baseEnv_conf)
        telem_v2 = collect(env, n_steps, action_fn, f"{test_name} [V2]", apply_patches=True)
        env.close()
        r_v2 = analyze(telem_v2, f"{test_name} [V2]")

        print_compare(test_name, r_orig, r_v2)
        all_results.append((test_name, r_orig, r_v2))

        tag = test_name.split(":")[0].strip().replace(" ", "_").lower()
        all_telem[f"{tag}_orig"] = telem_orig
        all_telem[f"{tag}_v2"]   = telem_v2

    # ── SAVE TELEMETRY ──
    telem_file = os.path.join(outdir, "benchmark_telemetry.json")
    with open(telem_file, 'w') as f:
        json.dump(all_telem, f, indent=2, default=str)
    print(f"\n📊 Full telemetry saved: {telem_file}")

    # ── FINAL SCORECARD ──
    print(f"\n{'='*70}")
    print(f"  FINAL SCORECARD: ORIGINAL vs V2")
    print(f"{'='*70}")

    v2_wins = 0
    orig_wins = 0
    ties = 0
    key_metrics = ['roll_std', 'roll_reversals', 'pitch_std', 'pitch_reversals',
                   'beta_max', 'yaw_rate_max']

    for test_name, r_orig, r_v2 in all_results:
        tw = 0; ow = 0
        for k in key_metrics:
            if r_v2.get(k, 0) < r_orig.get(k, 0): tw += 1
            elif r_v2.get(k, 0) > r_orig.get(k, 0): ow += 1
        if tw > ow:
            v2_wins += 1
            icon = "✅ V2"
        elif ow > tw:
            orig_wins += 1
            icon = "❌ ORIG"
        else:
            ties += 1
            icon = "🔄 TIE"
        print(f"  {icon}  {test_name}  (V2 better on {tw}/{len(key_metrics)} metrics)")

    print(f"\n  V2 wins: {v2_wins}  |  ORIG wins: {orig_wins}  |  Ties: {ties}")
    if v2_wins > orig_wins:
        print("  🎯 V2 AUTOPILOT IS SUPERIOR — READY FOR TRAINING!")
    elif v2_wins == orig_wins:
        print("  ⚖️  Results are mixed — may need further tuning.")
    else:
        print("  ⚠️  Original autopilot performed better — review V2 patches.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
