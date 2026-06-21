"""
collect_tactics_data.py — rich per-step instrumentation of the RL agent to
characterize LEARNED TACTICS & MANEUVERS. Logs bank, g-load, bearing-off-nose,
aspect, missile overload, and defensive evasion miss-distances.
  tactics/win_<k>.csv     full per-step for first wins (offensive tactics)
  tactics/evade_<k>.csv   full per-step for episodes surviving an incoming shot
  tactics/flyout_g.csv    missile Mach/alt/range/g over time
"""
import sys, os, csv, math
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.utils.geospatial import dinstance_between_agents, relative_bearing_between_agents
from jsb_gym.utils.wez import target_aspect_deg

MODEL='experiments/rir/rir_model_v3_best'; N_EP=80; OUT='tactics'
os.makedirs(OUT, exist_ok=True)
model=PPO.load(MODEL, device='cpu')

def gload(simobj):
    for p in ('accelerations/Nz','accelerations/n-pilot-z-norm','forces/load-factor'):
        try:
            v=simobj.fdm[p]
            if v is not None and abs(v)<50: return float(v)
        except Exception: pass
    return float('nan')

wins=evades=0
for ep in range(N_EP):
    if wins>=5 and evades>=3: break
    env=BVRBase(baseEnv_conf); obs,_=env.reset(seed=21000+ep)
    b,r=env.blue_agent,env.red_agent
    rows=[]; flyout=[]; prev_b=prev_r=False; red_miss_min=1e9; had_incoming=False
    done=trunc=False
    while not (done or trunc):
        t=b.simObj.get_sim_time_sec(); rng=dinstance_between_agents(b,r)/1e3
        b_active=b.is_own_missile_active(); r_active=r.is_own_missile_active()
        # track incoming-missile miss distance
        if r_active:
            had_incoming=True
            for m in r.ammo.values():
                if m.is_active() and m.target_norm: red_miss_min=min(red_miss_min,m.target_norm)
        rows.append(dict(t=round(t,1), range_km=round(rng,2),
            bearing_off_nose=round(abs(relative_bearing_between_agents(b,r)),1),
            aspect_deg=round(target_aspect_deg(b,r),1),
            heading=round(b.simObj.get_psi(),1), bank=round(b.simObj.get_phi(),1),
            pitch=round(b.simObj.get_theta(),1), alt_km=round(b.simObj.get_altitude()/1e3,3),
            mach=round(b.simObj.get_mach(),3), g=round(gload(b.simObj),2),
            blue_missile=int(b_active), red_missile=int(r_active)))
        if b_active:
            for m in b.ammo.values():
                if m.is_active():
                    flyout.append(dict(t=round(t,1), mach=round(m.get_mach(),3),
                        alt_km=round(m.get_altitude()/1e3,2),
                        dist_km=round((m.target_norm or 0)/1e3,2), g=round(gload(m),2)))
                    break
        prev_b,prev_r=b_active,r_active
        act,_=model.predict(obs,deterministic=True); obs,_,done,trunc,info=env.step(act)

    oc=info.get('outcome','?')
    if oc=='win' and wins<5:
        with open(f'{OUT}/win_{wins}.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if wins==0 and len(flyout)>5:
            with open(f'{OUT}/flyout_g.csv','w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(flyout[0].keys())); w.writeheader(); w.writerows(flyout)
        wins+=1
    # evasion episode: survived an incoming missile (alive, missile defeated, decent miss)
    if had_incoming and b.healthPoints>0 and evades<3 and red_miss_min<1e8:
        rows[-1]['miss_distance_m']=round(red_miss_min,1)
        with open(f'{OUT}/evade_{evades}.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"  evade_{evades}: miss distance {red_miss_min:,.0f} m, outcome {oc}")
        evades+=1

print(f"\nDONE: {wins} win logs, {evades} evade logs")
