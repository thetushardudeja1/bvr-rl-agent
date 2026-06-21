"""
Detailed missile telemetry profiler.

Instruments AAMBVR.step so EVERY internal sub-step of EVERY missile in an
engagement is recorded (distance, Mach, altitude, speed, heading, commanded
heading/altitude, along-LOS Doppler, notch/lost counters, guidance state, and
the exact frame/cause of death). Then writes per-missile CSVs, diagnostic plots,
and a written analysis aimed at the "weird terminal behaviour" question:
spiral/orbit detection, CPA vs effective_radius, boost/loft profile, and the
death cause for each shot.

Run:  python missile_telemetry.py [seed]
"""
import sys, os, math, csv
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.simObjects.missiles import AAMBVR
from jsb_gym.utils.geospatial import dinstance_between_simObj_agent

SEED  = int(sys.argv[1]) if len(sys.argv) > 1 else 7529
MODEL = 'experiments/rir/rir_notch2_best'
OUT   = '/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/SELF PLAY/missile_telemetry'
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Instrumentation: a logging replacement for AAMBVR.step that mirrors the real
# control flow but records the full state at each sub-step. Telemetry is keyed
# by missile python-id so blue and red shots stay separated, and tagged with a
# 'side' the first time we see each missile fire.
# ---------------------------------------------------------------------------
TELEM = {}        # id(missile) -> list[dict]
META  = {}        # id(missile) -> {'side':..., 'launch_dist':...}
SIDE  = {}        # id(missile) -> 'blue'/'red'  (filled in by the env loop)
_gt = {'t': 0.0}  # shared sim-time clock (advances by Sim*Control sub-frames)

def logged_step(self):
    mid = id(self)
    rec = TELEM.setdefault(mid, [])
    if mid not in META:
        META[mid] = {'launch_dist': None, 'apogee': 0.0}

    cfg = self.missile_simulation_config

    # --- coast (lock lost) phase: mirror _coast but log every frame ---
    if self.coasting:
        cap = getattr(self.missile_performance, 'coast_max_substeps', 240)
        for _ in range(int(cfg.Sim_time_step)):
            self.coast_steps += 1
            self.target_norm = dinstance_between_simObj_agent(self, self.target)
            self.is_mach_low(); self.is_alt_low()
            rec.append(dict(
                t=round(_gt['t'], 3), dist=self.target_norm,
                d_dist=(self.target_norm - self.target_norm_old) if self.target_norm_old is not None else 0.0,
                mach=self.get_mach(), tas=self.get_true_airspeed(), alt=self.get_altitude(),
                psi=self.get_psi(), theta=self.get_theta() if hasattr(self, 'get_theta') else float('nan'),
                v_down=self.get_v_down(), los_speed=float('nan'),
                notch_count=self.notch_count, lost_count=self.lost_count,
                hdg_cmd=getattr(self, 'last_heading_cmd', float('nan')),
                alt_cmd=getattr(self, 'last_altitude_cmd', float('nan')),
                boost_done=int(self.missile_control.acceleration_stage_done()),
                active=int(self.active), cause='coast'))
            self.target_norm_old = self.target_norm
            _gt['t'] += cfg.Control_time_step * self.conf.missile_navigation.dt
            if not self.alive:
                self.coasting = False; self.target_lost = True; return
            if self.coast_steps > cap:
                self.coasting = False; self.target_lost = True
                self.active = False; self.alive = False; return
            self._step_ballistic()
        return

    for _ in range(int(cfg.Sim_time_step)):
        self.target_norm = dinstance_between_simObj_agent(self, self.target)
        if META[mid]['launch_dist'] is None:
            META[mid]['launch_dist'] = self.target_norm

        # --- evaluate the SAME terminations as the live step(), in order ---
        self.is_mach_low()
        self.is_alt_low()
        cause = None
        if self.active:
            if self.is_target_within_effective_range():
                cause = 'direct_hit'
        if self.active:
            if self.is_detonated_at_cpa():
                cause = 'cpa_hit' if self.target_hit else 'cpa_miss'
        if self.active:
            if self.is_target_lost():
                cause = 'lost_opening'
        if self.active:
            if self.is_notched():
                cause = 'notched'

        # --- record state for this sub-step ---
        try:
            los = self.target_los_speed()
        except Exception:
            los = float('nan')
        alt = self.get_altitude()
        META[mid]['apogee'] = max(META[mid]['apogee'], alt)
        rec.append(dict(
            t=round(_gt['t'], 3),
            dist=self.target_norm,
            d_dist=(self.target_norm - self.target_norm_old) if self.target_norm_old is not None else 0.0,
            mach=self.get_mach(),
            tas=self.get_true_airspeed(),
            alt=alt,
            psi=self.get_psi(),
            theta=self.get_theta() if hasattr(self, 'get_theta') else float('nan'),
            v_down=self.get_v_down(),
            los_speed=los,
            notch_count=self.notch_count,
            lost_count=self.lost_count,
            hdg_cmd=getattr(self, 'last_heading_cmd', float('nan')),
            alt_cmd=getattr(self, 'last_altitude_cmd', float('nan')),
            boost_done=int(self.missile_control.acceleration_stage_done()),
            active=int(self.active),
            cause=cause or '',
        ))
        self.target_norm_old = self.target_norm
        _gt['t'] += cfg.Control_time_step * self.conf.missile_navigation.dt

        if not self.active:
            return
        self._step()

AAMBVR.step = logged_step

# ---------------------------------------------------------------------------
# Run one engagement
# ---------------------------------------------------------------------------
baseEnv_conf.red_doctrine_randomize = False
model = PPO.load(MODEL, device='cpu')
env = BVRBase(baseEnv_conf)
obs, _ = env.reset(seed=SEED)
done = trunc = False
while not (done or trunc):
    # tag any newly-fired missiles with their side
    for m in env.blue_agent.ammo.values():
        if m.is_active() or id(m) in TELEM:
            SIDE[id(m)] = 'blue'
    for m in env.red_agent.ammo.values():
        if m.is_active() or id(m) in TELEM:
            SIDE[id(m)] = 'red'
    a, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, info = env.step(a)
outcome = info.get('outcome', '?')

# ---------------------------------------------------------------------------
# Analysis + CSV + plots
# ---------------------------------------------------------------------------
def turn_rate_series(psi):
    out = [0.0]
    for i in range(1, len(psi)):
        d = psi[i] - psi[i-1]
        while d > 180: d -= 360
        while d < -180: d += 360
        out.append(d)
    return out

print(f"\n{'='*70}\n MISSILE TELEMETRY  seed={SEED}  outcome={outcome.upper()}\n{'='*70}")

shots = [(mid, recs) for mid, recs in TELEM.items() if recs]
shots.sort(key=lambda kv: kv[1][0]['t'])
report_lines = [f"MISSILE TELEMETRY ANALYSIS  seed={SEED}  outcome={outcome}", ""]

for n, (mid, recs) in enumerate(shots, 1):
    side = SIDE.get(mid, '?')
    dists = [r['dist'] for r in recs]
    machs = [r['mach'] for r in recs]
    alts  = [r['alt'] for r in recs]
    psis  = [r['psi'] for r in recs]
    ts    = [r['t'] for r in recs]
    trate = turn_rate_series(psis)
    cpa_i = int(np.argmin(dists))
    cpa   = dists[cpa_i]
    cause = next((r['cause'] for r in recs if r['cause']), 'still_flying/timeout')

    # spiral metric: total heading change AFTER closest approach (an orbiting
    # missile keeps turning hard once it has overshot)
    post = trate[cpa_i:]
    post_turn = sum(abs(x) for x in post)
    max_turn_rate = max((abs(x) for x in trate), default=0.0)

    # write CSV
    csv_path = os.path.join(OUT, f"seed{SEED}_{side}_shot{n}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)

    block = [
        f"--- {side.upper()} shot #{n}  ({len(recs)} sub-steps, {ts[-1]-ts[0]:.1f}s flight) ---",
        f"  launch range     : {META[mid]['launch_dist']/1e3:7.2f} km",
        f"  apogee (loft)    : {META[mid]['apogee']/1e3:7.2f} km   launch alt {alts[0]/1e3:.2f} km",
        f"  peak Mach        : {max(machs):6.2f}   terminal Mach {machs[cpa_i]:.2f}",
        f"  closest approach : {cpa:7.1f} m  (effective_radius {recs[0] and env.red_agent.ammo['0'].missile_performance.effective_radius:.0f} m)",
        f"  terminal closing : {min(r['d_dist'] for r in recs):7.1f} m/sub-step (most negative = fastest closing)",
        f"  death cause      : {cause}",
        f"  max turn rate    : {max_turn_rate:6.2f} deg/sub-step",
        f"  post-CPA turning : {post_turn:6.1f} deg total  -> {'>>> SPIRAL/ORBIT' if post_turn > 180 else 'clean (no orbit)'}",
        f"  CSV              : {os.path.basename(csv_path)}",
        "",
    ]
    print("\n".join(block))
    report_lines += block

# plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for mid, recs in shots:
        side = SIDE.get(mid, '?')
        ts = [r['t'] for r in recs]
        c = 'tab:blue' if side == 'blue' else 'tab:red'
        ax[0,0].plot(ts, [r['dist']/1e3 for r in recs], color=c, label=side)
        ax[0,1].plot(ts, [r['mach'] for r in recs], color=c)
        ax[1,0].plot(ts, [r['alt']/1e3 for r in recs], color=c)
        ax[1,1].plot(ts, [r['los_speed'] for r in recs], color=c)
    ax[0,0].set_title('Distance to target (km)'); ax[0,0].axhline(0.3, ls=':', c='k', lw=.8); ax[0,0].legend()
    ax[0,1].set_title('Mach');            ax[0,1].axhline(1.0, ls=':', c='k', lw=.8)
    ax[1,0].set_title('Altitude (km)')
    ax[1,1].set_title('Along-LOS Doppler (m/s)'); ax[1,1].axhline(75, ls=':', c='k', lw=.8)
    for a in ax.flat: a.set_xlabel('time (s)'); a.grid(alpha=.3)
    fig.suptitle(f'Missile telemetry  seed={SEED}  outcome={outcome}')
    fig.tight_layout()
    png = os.path.join(OUT, f'seed{SEED}_profile.png')
    fig.savefig(png, dpi=110)
    print(f"plots -> {png}")
    report_lines.append(f"plots -> {os.path.basename(png)}")
except Exception as e:
    print(f"(plot skipped: {e})")

with open(os.path.join(OUT, f'seed{SEED}_analysis.txt'), 'w') as f:
    f.write("\n".join(report_lines))
print(f"\nAll telemetry written to: {OUT}")
