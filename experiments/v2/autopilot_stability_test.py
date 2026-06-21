"""
autopilot_stability_test.py
============================
Exhaustive stability & wobble diagnostic for the V2 PID autopilot.
Runs 5 flight regimes, logs full telemetry every step, and produces
a summary report with pass/fail criteria.

All V2 patches are applied IN MEMORY after env.reset() — core code untouched.

Architecture reminder:
  PPO → [heading, altitude, throttle]
    → F16BVR._step() computes diff_head, diff_alt
      → AircraftPIDAutopilot.get_control_input(diff_head, diff_alt)
        → returns (aileron_cmd, elevator_cmd, rudder_cmd=0.0 always!)
      → F16BVR.command_aircraft() sends to JSBSim FDM
        → fdm.run() × 10 inner steps × 6 outer steps = 60 FDM steps per RL step
"""

import os, sys, math, json
import numpy as np

ROOT = os.path.expanduser("~/bvrgym_project_new")
sys.path.insert(0, ROOT)

from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf


def apply_v2_patches(env):
    """
    Apply V2 autopilot fixes to blue aircraft AFTER env.reset().
    Patches are applied to the live object instances, not the classes.
    """
    blue_aircraft = env.blue_agent.simObj  # F16BVR instance
    autopilot = blue_aircraft.aircraft_control  # AircraftPIDAutopilot instance
    fdm = blue_aircraft.fdm  # JSBSim FDM

    # ── FIX 1: Reduce Roll_max from 80° to 35° ──
    blue_aircraft.conf.aircraft_navigation.Roll_max = 35

    # ── FIX 2: Remove pitch PID D-term (causes porpoising) ──
    # Pitch PID has D=1.0 which makes the nose bounce
    autopilot.pitch_PID.Kd = 0.0
    autopilot.pitch_PID.Derivator = 0.0

    # ── FIX 3: Add yaw damper by wrapping command_aircraft ──
    # The original code sets rudder_cmd = 0.0 ALWAYS — no yaw coordination!
    blue_aircraft._yaw_rate_prev = 0.0
    blue_aircraft._yaw_integral = 0.0

    _orig_command = blue_aircraft.command_aircraft

    def _patched_command(aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd):
        # Read yaw rate from FDM
        try:
            yaw_rate = fdm['velocities/r-rad_sec']
            # PD yaw damper
            rudder_damp = -1.5 * yaw_rate - 0.3 * (yaw_rate - blue_aircraft._yaw_rate_prev)
            # Small integral to zero out steady-state sideslip
            blue_aircraft._yaw_integral += yaw_rate * 0.0833
            blue_aircraft._yaw_integral = max(-0.5, min(0.5, blue_aircraft._yaw_integral))
            rudder_damp -= 0.2 * blue_aircraft._yaw_integral
            rudder_cmd = max(-1.0, min(1.0, rudder_damp))
            blue_aircraft._yaw_rate_prev = yaw_rate
        except:
            pass

        # Roll limiter: if bank angle exceeds 35°, apply corrective aileron
        try:
            phi = fdm['attitude/phi-rad']
            max_roll = math.radians(35)
            if abs(phi) > max_roll:
                correction = -0.5 * (phi - math.copysign(max_roll, phi))
                aileron_cmd = max(-1, min(1, aileron_cmd + correction))
        except:
            pass

        _orig_command(aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd)

    blue_aircraft.command_aircraft = _patched_command

    # ── FIX 4: Wrap _step to reset integrators on large target jumps ──
    blue_aircraft._prev_hdg_target = None
    blue_aircraft._prev_alt_target = None
    _orig_step = blue_aircraft._step

    def _patched_step(action):
        hdg_t = action[0]
        alt_t = action[1]
        # Reset integrators on large command changes
        if blue_aircraft._prev_hdg_target is not None:
            from jsb_gym.utils.navigation import delta_heading
            dh = abs(delta_heading(hdg_t, blue_aircraft._prev_hdg_target))
            if dh > 15:
                autopilot.roll_PID.Integrator = 0.0
                autopilot.roll_PID.Derivator = 0.0
                autopilot.roll_sec_PID.Integrator = 0.0
        if blue_aircraft._prev_alt_target is not None:
            if abs(alt_t - blue_aircraft._prev_alt_target) > 500:
                autopilot.pitch_PID.Integrator = 0.0
                autopilot.pitch_PID.Derivator = 0.0
        blue_aircraft._prev_hdg_target = hdg_t
        blue_aircraft._prev_alt_target = alt_t
        _orig_step(action)

    blue_aircraft._step = _patched_step

    print("  [V2] Patches applied: Roll_max=35°, Pitch D=0, Yaw Damper ON, Integrator Resets ON")


def collect_telemetry(env, n_steps, action_fn, label):
    """Run n_steps, calling action_fn(step_i, obs) each step. Log full FDM telemetry."""
    obs = env.reset()[0]
    apply_v2_patches(env)

    telemetry = []
    fdm = env.blue_agent.simObj.fdm

    for step_i in range(n_steps):
        action = action_fn(step_i, obs)
        obs, reward, done, trunc, info = env.step(action)

        try:
            row = {
                'step': step_i,
                'label': label,
                # Position
                'alt_ft':      fdm['position/h-sl-ft'],
                # Attitude (degrees)
                'roll_deg':    math.degrees(fdm['attitude/phi-rad']),
                'pitch_deg':   math.degrees(fdm['attitude/theta-rad']),
                'heading_deg': math.degrees(fdm['attitude/psi-rad']),
                # Angular rates (deg/s)
                'roll_rate_dps':  math.degrees(fdm['velocities/p-rad_sec']),
                'pitch_rate_dps': math.degrees(fdm['velocities/q-rad_sec']),
                'yaw_rate_dps':   math.degrees(fdm['velocities/r-rad_sec']),
                # Speed
                'airspeed_kt': fdm['velocities/vc-kts'],
                'mach':        fdm['velocities/mach'],
                # Control surfaces
                'aileron_cmd': fdm['fcs/aileron-cmd-norm'],
                'elevator_cmd':fdm['fcs/elevator-cmd-norm'],
                'rudder_cmd':  fdm['fcs/rudder-cmd-norm'],
                # Aero
                'alpha_deg':   math.degrees(fdm['aero/alpha-rad']),
                'beta_deg':    math.degrees(fdm['aero/beta-rad']),
                'g_load':      fdm['accelerations/Nz'],
            }
        except Exception as e:
            row = {'step': step_i, 'label': label, 'error': str(e)}

        telemetry.append(row)

        if done or trunc:
            print(f"    Episode ended early at step {step_i}: done={done}, trunc={trunc}")
            break

    return telemetry


def analyze(telem, label):
    """Compute stability metrics."""
    n = len(telem)
    if n < 2:
        return {'label': label, 'steps': n, 'error': 'too few'}

    def col(key):
        return [t.get(key, 0) for t in telem]

    rolls = col('roll_deg')
    pitches = col('pitch_deg')
    headings = col('heading_deg')
    alts = col('alt_ft')
    yaw_rates = col('yaw_rate_dps')
    betas = col('beta_deg')
    g_loads = col('g_load')
    airspeeds = col('airspeed_kt')
    roll_rates = col('roll_rate_dps')
    pitch_rates = col('pitch_rate_dps')

    # Per-step deltas
    d_roll = [abs(rolls[i] - rolls[i-1]) for i in range(1, n)]
    d_pitch = [abs(pitches[i] - pitches[i-1]) for i in range(1, n)]
    d_hdg = [min(abs(headings[i] - headings[i-1]),
                 360 - abs(headings[i] - headings[i-1])) for i in range(1, n)]

    # Sign reversals (oscillation indicator)
    roll_rev = sum(1 for i in range(2, n)
                   if (rolls[i]-rolls[i-1]) * (rolls[i-1]-rolls[i-2]) < 0)
    pitch_rev = sum(1 for i in range(2, n)
                    if (pitches[i]-pitches[i-1]) * (pitches[i-1]-pitches[i-2]) < 0)

    return {
        'label': label, 'steps': n,
        # ── ROLL ──
        'roll_mean':       round(np.mean(rolls), 2),
        'roll_std':        round(np.std(rolls), 2),
        'roll_max_abs':    round(max(abs(r) for r in rolls), 2),
        'roll_Δ_mean':     round(np.mean(d_roll), 3),
        'roll_Δ_max':      round(max(d_roll), 3),
        'roll_reversals':  roll_rev,
        'roll_rate_max':   round(max(abs(r) for r in roll_rates), 2),
        # ── PITCH ──
        'pitch_mean':      round(np.mean(pitches), 2),
        'pitch_std':       round(np.std(pitches), 2),
        'pitch_Δ_mean':    round(np.mean(d_pitch), 3),
        'pitch_Δ_max':     round(max(d_pitch), 3),
        'pitch_reversals': pitch_rev,
        'pitch_rate_max':  round(max(abs(r) for r in pitch_rates), 2),
        # ── HEADING ──
        'hdg_Δ_mean':      round(np.mean(d_hdg), 3),
        'hdg_Δ_max':       round(max(d_hdg), 3),
        # ── ALTITUDE ──
        'alt_mean':        round(np.mean(alts), 1),
        'alt_std':         round(np.std(alts), 1),
        'alt_min':         round(min(alts), 1),
        'alt_max':         round(max(alts), 1),
        'alt_drift':       round(alts[-1] - alts[0], 1),
        # ── YAW / SIDESLIP (Dutch roll) ──
        'yaw_rate_mean':   round(np.mean([abs(y) for y in yaw_rates]), 3),
        'yaw_rate_max':    round(max(abs(y) for y in yaw_rates), 3),
        'beta_mean':       round(np.mean([abs(b) for b in betas]), 3),
        'beta_max':        round(max(abs(b) for b in betas), 3),
        # ── G-LOAD ──
        'g_mean':          round(np.mean(g_loads), 3),
        'g_max':           round(max(abs(g) for g in g_loads), 3),
        # ── SPEED ──
        'speed_mean':      round(np.mean(airspeeds), 1),
        'speed_min':       round(min(airspeeds), 1),
    }


def grade(report, test_type):
    """Pass/fail with clear criteria."""
    issues = []
    if test_type == 'straight':
        if report['roll_std'] > 3.0:
            issues.append(f"ROLL STD {report['roll_std']}° > 3.0° limit")
        if report['alt_std'] > 300:
            issues.append(f"ALT STD {report['alt_std']}ft > 300ft limit")
        if report['beta_max'] > 5.0:
            issues.append(f"SIDESLIP β={report['beta_max']}° > 5.0° (Dutch roll)")
        if report['roll_reversals'] > 30:
            issues.append(f"ROLL REVERSALS {report['roll_reversals']} > 30 (oscillating)")
        if report['pitch_reversals'] > 30:
            issues.append(f"PITCH REVERSALS {report['pitch_reversals']} > 30 (porpoising)")
    elif test_type == 'turn':
        if report['roll_max_abs'] > 42:
            issues.append(f"ROLL {report['roll_max_abs']}° > 42° (exceeded limit)")
        if report['beta_max'] > 10:
            issues.append(f"SIDESLIP β={report['beta_max']}° > 10° (Dutch roll in turn)")
        if report['roll_reversals'] > 50:
            issues.append(f"ROLL REVERSALS {report['roll_reversals']} > 50 (wobble in turn)")
    elif test_type == 'altitude':
        if report['pitch_reversals'] > 60:
            issues.append(f"PITCH REVERSALS {report['pitch_reversals']} > 60 (porpoising)")
        if report['g_max'] > 6.0:
            issues.append(f"G-LOAD {report['g_max']}g > 6.0g (structural)")
    elif test_type == 'rapid':
        if report['roll_Δ_mean'] > 10.0:
            issues.append(f"ROLL JITTER {report['roll_Δ_mean']}°/step > 10° limit")
        if report['pitch_Δ_mean'] > 8.0:
            issues.append(f"PITCH JITTER {report['pitch_Δ_mean']}°/step > 8° limit")
        if report['speed_min'] < 100:
            issues.append(f"STALL RISK: {report['speed_min']}kt < 100kt")
    return ('PASS' if not issues else 'FAIL', issues)


def print_report(report, test_type):
    status, issues = grade(report, test_type)
    icon = "✅" if status == 'PASS' else "❌"
    print(f"  {icon} {report['label']}: {status}")
    print(f"      Roll:  σ={report['roll_std']}°  max={report['roll_max_abs']}°  "
          f"Δ/step={report['roll_Δ_mean']}°  reversals={report['roll_reversals']}  "
          f"rate_max={report['roll_rate_max']}°/s")
    print(f"      Pitch: σ={report['pitch_std']}°  Δ/step={report['pitch_Δ_mean']}°  "
          f"reversals={report['pitch_reversals']}  rate_max={report['pitch_rate_max']}°/s")
    print(f"      Yaw:   rate_max={report['yaw_rate_max']}°/s  β_max={report['beta_max']}°")
    print(f"      Alt:   σ={report['alt_std']}ft  range=[{report['alt_min']}, {report['alt_max']}]ft  "
          f"drift={report['alt_drift']}ft")
    print(f"      Speed: mean={report['speed_mean']}kt  min={report['speed_min']}kt  "
          f"G_max={report['g_max']}g")
    for iss in issues:
        print(f"      ⚠️  {iss}")
    return status, issues


# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  V2 AUTOPILOT STABILITY TEST SUITE")
    print("  Exhaustive telemetry & wobble diagnostic")
    print("=" * 70)

    baseEnv_conf.tacview_output = False
    outdir = os.path.join(ROOT, "experiments", "v2")
    os.makedirs(outdir, exist_ok=True)
    all_results = []
    all_telem = {}

    # ─── TEST 1: Straight & Level (200 steps) ─────────────────────
    print("\n── TEST 1: STRAIGHT & LEVEL HOLD (200 steps) ──")
    env = BVRBase(baseEnv_conf)
    def straight_action(step, obs):
        return np.array([0.0, 0.0, 0.5])  # no heading/alt change, half throttle
    telem = collect_telemetry(env, 200, straight_action, "Straight & Level")
    env.close()
    r = analyze(telem, "TEST 1: Straight & Level")
    s, i = print_report(r, 'straight')
    r['status'] = s; r['issues'] = i
    all_results.append(r)
    all_telem['straight_level'] = telem

    # ─── TEST 2: 90° Right Turn (150 steps) ───────────────────────
    print("\n── TEST 2: 90° RIGHT TURN (150 steps) ──")
    env = BVRBase(baseEnv_conf)
    def right_turn(step, obs):
        return np.array([0.7, 0.0, 0.7])  # strong right heading bias
    telem = collect_telemetry(env, 150, right_turn, "90° Right Turn")
    env.close()
    r = analyze(telem, "TEST 2: 90° Right Turn")
    s, i = print_report(r, 'turn')
    r['status'] = s; r['issues'] = i
    all_results.append(r)
    all_telem['right_turn'] = telem

    # ─── TEST 3: 90° Left Turn (150 steps) ────────────────────────
    print("\n── TEST 3: 90° LEFT TURN (150 steps) ──")
    env = BVRBase(baseEnv_conf)
    def left_turn(step, obs):
        return np.array([-0.7, 0.0, 0.7])
    telem = collect_telemetry(env, 150, left_turn, "90° Left Turn")
    env.close()
    r = analyze(telem, "TEST 3: 90° Left Turn")
    s, i = print_report(r, 'turn')
    r['status'] = s; r['issues'] = i
    all_results.append(r)
    all_telem['left_turn'] = telem

    # ─── TEST 4: Climb then Descend (200 steps) ───────────────────
    print("\n── TEST 4: CLIMB THEN DESCEND (200 steps) ──")
    env = BVRBase(baseEnv_conf)
    def climb_descend(step, obs):
        if step < 100:
            return np.array([0.0, 0.5, 0.8])   # climb
        else:
            return np.array([0.0, -0.5, 0.5])   # descend
    telem = collect_telemetry(env, 200, climb_descend, "Climb & Descend")
    env.close()
    r = analyze(telem, "TEST 4: Climb & Descend")
    s, i = print_report(r, 'altitude')
    r['status'] = s; r['issues'] = i
    all_results.append(r)
    all_telem['climb_descend'] = telem

    # ─── TEST 5: Rapid Random Commands (200 steps, RL-like) ───────
    print("\n── TEST 5: RAPID RANDOM COMMANDS (200 steps) ──")
    env = BVRBase(baseEnv_conf)
    rng = np.random.RandomState(42)
    def rapid_random(step, obs):
        a = rng.uniform(-1, 1, size=3)
        a[2] = abs(a[2])  # throttle positive
        return a
    telem = collect_telemetry(env, 200, rapid_random, "Rapid Random")
    env.close()
    r = analyze(telem, "TEST 5: Rapid Random Commands")
    s, i = print_report(r, 'rapid')
    r['status'] = s; r['issues'] = i
    all_results.append(r)
    all_telem['rapid_random'] = telem

    # ─── SAVE RAW TELEMETRY ───────────────────────────────────────
    telem_file = os.path.join(outdir, "v2_stability_telemetry.json")
    with open(telem_file, 'w') as f:
        json.dump(all_telem, f, indent=2, default=str)
    print(f"\n  📊 Full telemetry saved: {telem_file}")

    # ─── FINAL SUMMARY ───────────────────────────────────────────
    passed = sum(1 for r in all_results if r['status'] == 'PASS')
    total = len(all_results)

    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)
    for r in all_results:
        icon = "✅" if r['status'] == 'PASS' else "❌"
        print(f"  {icon} {r['label']}: {r['status']}")

    print(f"\n  OVERALL: {passed}/{total} tests passed")
    if passed == total:
        print("  🎯 V2 AUTOPILOT CERTIFIED STABLE — READY FOR TRAINING!")
    else:
        print("  ⚠️  Review failed tests before starting training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
