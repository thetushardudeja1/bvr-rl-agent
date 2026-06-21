"""
generate_proof_flight.py
========================
One deterministic bc_model_v3-vs-BT episode producing:
  1. data_output/tacview/BVRGym_engagement.acmi  (open in Tacview)
  2. data_output/flight_telemetry.csv            (per-control-tick blue state)
  3. stdout bang-bang / stability analysis of the telemetry

Telemetry is sampled every agent step (0.5s) from JSBSim state; the analysis
quantifies the old corkscrew signature: bank reversals, roll-rate, command
slamming.
"""
import sys, os, csv, math
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.navigation import delta_heading

model = PPO.load('experiments/rir/bc_model_v3', device='cpu')
env = BVRBase(baseEnv_conf)
obs, _ = env.reset(seed=42)

rows = []
done = trunc = False
while not (done or trunc):
    env.log_tacview()                       # builds the ACMI frame by frame
    act, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, info = env.step(act)
    b = env.blue_agent.simObj
    rows.append(dict(
        t=b.get_sim_time_sec(), phi=b.get_phi(), theta=b.get_theta(),
        psi=b.get_psi(), alt=b.get_altitude(), mach=b.get_mach(),
        cmd_heading=float(act[0]), cmd_alt=float(act[1]),
        launch=float(act[3]) if len(act) > 3 else 0.0,
        aileron=b.fdm['fcs/aileron-cmd-norm'],
        elevator=b.fdm['fcs/elevator-cmd-norm'],
    ))
env.done = True
env.log_tacview()                            # triggers save_logs()

os.makedirs('data_output', exist_ok=True)
with open('data_output/flight_telemetry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---------------- bang-bang / stability analysis ----------------
phi  = np.array([r['phi'] for r in rows])
ail  = np.array([r['aileron'] for r in rows])
elev = np.array([r['elevator'] for r in rows])
t    = np.array([r['t'] for r in rows])
dt   = np.diff(t)
roll_rate = np.abs(np.diff(phi)) / np.maximum(dt, 1e-6)   # deg/s (0.5s sampled)

# bank reversal: sign flip of phi with both sides beyond 15 deg (corkscrew move)
sign = np.sign(phi)
reversals = int(np.sum((sign[1:] != sign[:-1]) &
                       (np.abs(phi[1:]) > 15) & (np.abs(phi[:-1]) > 15)))
minutes = (t[-1] - t[0]) / 60.0

# command slamming: fraction of time surfaces are at >90% deflection
ail_slam  = float((np.abs(ail)  > 0.9).mean())
elev_slam = float((np.abs(elev) > 0.9).mean())

print("\n================ FLIGHT STABILITY REPORT ================")
print(f"  episode      : {minutes:.1f} min, outcome = {info.get('outcome','?')}")
print(f"  bank angle   : mean|phi| {np.mean(np.abs(phi)):6.1f}  max|phi| {np.max(np.abs(phi)):6.1f} deg")
print(f"  bank reversal: {reversals} total = {reversals/max(minutes,1e-6):.1f}/min  (corkscrew was ~continuous)")
print(f"  roll rate    : mean {np.mean(roll_rate):6.1f}  p95 {np.percentile(roll_rate,95):6.1f} deg/s")
print(f"  aileron |>0.9|: {ail_slam:6.1%} of samples   (bang-bang slammed constantly)")
print(f"  elevator|>0.9|: {elev_slam:6.1%} of samples  (old code hard-coded -0.9)")
print(f"  altitude     : min {min(r['alt'] for r in rows):,.0f}  max {max(r['alt'] for r in rows):,.0f} m")
print(f"  launches cmd : {sum(1 for i in range(1,len(rows)) if rows[i]['launch']>0 and rows[i-1]['launch']<=0)}")
print("==========================================================")
print("  ACMI  -> data_output/tacview/BVRGym_engagement.acmi")
print("  CSV   -> data_output/flight_telemetry.csv")
