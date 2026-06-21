"""
collect_analysis_data.py — instrument the RL agent (vs BT, deterministic) to
produce the data behind a BVR-grade figure set. Logs:
  analysis/episodes.csv  — per episode: initial geometry + outcome + kill range
  analysis/launches.csv  — every Blue missile launch: range/aspect/alt/mach
  analysis/detail_<k>.csv— full per-step state for the first few WINS
  analysis/flyout.csv    — a missile's Mach/altitude/range vs time
"""
import sys, os, csv, math
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents
from jsb_gym.utils.wez import target_aspect_deg

MODEL = 'experiments/rir/rir_model_v3_best'
N_EP = 150
G = 9.81
OUT = 'analysis'
os.makedirs(OUT, exist_ok=True)
model = PPO.load(MODEL, device='cpu')

def specific_energy(simobj):
    v = simobj.get_true_airspeed()
    return simobj.get_altitude() + v*v/(2*G)

ep_rows, launch_rows = [], []
detail_saved, flyout_saved = 0, False

for ep in range(N_EP):
    env = BVRBase(baseEnv_conf)
    obs, _ = env.reset(seed=12000 + ep)
    b, r = env.blue_agent, env.red_agent
    init_range = dinstance_between_agents(b, r) / 1e3
    init_aspect = target_aspect_deg(b, r)
    init_balt = b.simObj.get_altitude() / 1e3
    init_ralt = r.simObj.get_altitude() / 1e3

    detail = []          # this episode's per-step state
    flyout = []          # first missile's flyout
    blue_shots = red_shots = 0
    prev_b = prev_r = False
    prev_range = init_range
    kill_range = float('nan')
    done = trunc = False

    while not (done or trunc):
        t = b.simObj.get_sim_time_sec()
        rng = dinstance_between_agents(b, r) / 1e3
        asp = target_aspect_deg(b, r)
        closure = (prev_range - rng) * 1000 / 0.5   # m/s
        b_active = b.is_own_missile_active()
        r_active = r.is_own_missile_active()

        # launch event (Blue)
        if b_active and not prev_b:
            blue_shots += 1
            launch_rows.append(dict(ep=ep, t=round(t,1), range_km=round(rng,2),
                                    aspect_deg=round(asp,1),
                                    blue_alt_km=round(b.simObj.get_altitude()/1e3,2),
                                    blue_mach=round(b.simObj.get_mach(),3)))
        if r_active and not prev_r:
            red_shots += 1
        # kill range = range when Red dies
        if r.healthPoints <= 0 and math.isnan(kill_range):
            kill_range = rng

        detail.append(dict(t=round(t,1), range_km=round(rng,3),
                           closure_mps=round(closure,1), aspect_deg=round(asp,1),
                           blue_alt=round(b.simObj.get_altitude(),1),
                           red_alt=round(r.simObj.get_altitude(),1),
                           blue_mach=round(b.simObj.get_mach(),3),
                           red_mach=round(r.simObj.get_mach(),3),
                           blue_Es=round(specific_energy(b.simObj),1),
                           red_Es=round(specific_energy(r.simObj),1),
                           blue_missile=int(b_active), red_missile=int(r_active)))

        # flyout: sample the first Blue missile in flight
        if not flyout_saved and b_active:
            for m in b.ammo.values():
                if m.is_active():
                    flyout.append(dict(t=round(t,1),
                                       missile_mach=round(m.get_mach(),3),
                                       missile_alt=round(m.get_altitude(),1),
                                       dist_km=round((m.target_norm or 0)/1e3,2)))
                    break

        prev_b, prev_r, prev_range = b_active, r_active, rng
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(act)

    outcome = info.get('outcome', '?')
    dur = b.simObj.get_sim_time_sec()
    ep_rows.append(dict(ep=ep, init_range_km=round(init_range,1),
                        init_aspect_deg=round(init_aspect,1),
                        init_blue_alt_km=round(init_balt,1),
                        init_red_alt_km=round(init_ralt,1),
                        alt_adv_km=round(init_balt-init_ralt,1),
                        outcome=outcome, duration_s=round(dur,1),
                        kill_range_km=round(kill_range,2) if not math.isnan(kill_range) else '',
                        blue_shots=blue_shots, red_shots=red_shots))

    if outcome == 'win' and detail_saved < 3:
        with open(f'{OUT}/detail_{detail_saved}.csv','w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(detail[0].keys())); w.writeheader(); w.writerows(detail)
        detail_saved += 1
    if not flyout_saved and len(flyout) > 5:
        with open(f'{OUT}/flyout.csv','w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(flyout[0].keys())); w.writeheader(); w.writerows(flyout)
        flyout_saved = True
    if (ep+1) % 25 == 0:
        wins = sum(1 for e in ep_rows if e['outcome']=='win')
        print(f"  ep {ep+1}/{N_EP}  win-rate so far {wins/len(ep_rows):.0%}")

with open(f'{OUT}/episodes.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys())); w.writeheader(); w.writerows(ep_rows)
with open(f'{OUT}/launches.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(launch_rows[0].keys())); w.writeheader(); w.writerows(launch_rows)

wins = sum(1 for e in ep_rows if e['outcome']=='win')
print(f"\nDONE: {N_EP} eps, win-rate {wins/N_EP:.1%}, {len(launch_rows)} launches, "
      f"{detail_saved} detail logs, flyout={'yes' if flyout_saved else 'no'}")
