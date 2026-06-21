"""
make_tactics_figures.py — LEARNED TACTICS & MANEUVERS figure pack.
Reads tactics/*.csv from collect_tactics_data.py.
"""
import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi':130,'font.size':11,'axes.grid':True,'grid.alpha':0.3,'axes.axisbelow':True})
TC=os.path.expanduser('~/bvr_analysis/tactics'); OUT=os.path.expanduser('~/bvr_v2_clean/figures')
os.makedirs(OUT,exist_ok=True)
BLUE,RED,GREEN,ORANGE,PURPLE='#1f6fc4','#c0392b','#27ae60','#e67e22','#8e44ad'

def load(p):
    rows=list(csv.DictReader(open(p)))
    cols={k:[] for k in rows[0]}
    for r in rows:
        for k,v in r.items():
            try: cols[k].append(float(v))
            except: cols[k].append(np.nan)
    return {k:np.array(v) for k,v in cols.items()}

def launch_time(d):
    bm=d['blue_missile']; idx=np.where(bm>0)[0]
    return d['t'][idx[0]] if len(idx) else None

# T1 ── annotated engagement phases
def t_phases():
    p=f'{TC}/win_0.csv'
    if not os.path.exists(p): print('  [skip] phases'); return
    d=load(p); t=d['t']; lt=launch_time(d)
    fig,ax=plt.subplots(figsize=(11,5))
    ax.plot(t,d['range_km'],color=BLUE,lw=2,label='range')
    ax2=ax.twinx(); ax2.plot(t,d['bearing_off_nose'],color=ORANGE,lw=1.3,alpha=0.8,label='bearing off nose')
    ax.set_xlabel('time (s)'); ax.set_ylabel('range to enemy (km)',color=BLUE)
    ax2.set_ylabel('bearing off nose (deg)',color=ORANGE)
    if lt is not None:
        ax.axvline(lt,color=RED,ls='--',lw=1.5); ax.text(lt,ax.get_ylim()[1]*0.92,' LAUNCH',color=RED,fontweight='bold')
    # phase shading
    tmax=t[-1]
    phases=[(0,lt or tmax*0.4,'#d6eaf8','APPROACH\n(close & aspect)'),
            (lt or tmax*0.4,(lt or tmax*0.4)+30,'#fdebd0','CRANK/SUPPORT'),
            ((lt or tmax*0.4)+30,tmax,'#d5f5e3','TERMINAL / re-engage')]
    for x0,x1,c,lbl in phases:
        ax.axvspan(x0,x1,color=c,alpha=0.4); ax.text((x0+x1)/2,ax.get_ylim()[1]*0.05,lbl,ha='center',fontsize=8)
    ax.set_title('Anatomy of a BVR attack — learned engagement phases',fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{OUT}/T1_engagement_phases.png'); plt.close(fig); print('  T1_engagement_phases.png')

# T2 ── crank: heading change after launch while keeping target in cone
def t_crank():
    p=f'{TC}/win_0.csv'
    if not os.path.exists(p): return
    d=load(p); t=d['t']; lt=launch_time(d)
    if lt is None: print('  [skip] crank'); return
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.plot(t,d['bearing_off_nose'],color=BLUE,lw=1.8,label='bearing off nose (target position)')
    ax.plot(t,np.abs(d['bank']),color=PURPLE,lw=1.3,alpha=0.8,label='|bank angle|')
    ax.axvline(lt,color=RED,ls='--',lw=1.5,label='missile launch')
    ax.axhspan(0,40,color=GREEN,alpha=0.12); ax.text(t[-1],38,'40° seeker cone',ha='right',color=GREEN,fontsize=9)
    ax.set_xlabel('time (s)'); ax.set_ylabel('degrees')
    ax.set_title('Crank maneuver — bank away after launch while keeping target in seeker cone',fontweight='bold')
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(f'{OUT}/T2_crank.png'); plt.close(fig); print('  T2_crank.png')

# T3 ── defensive evasion
def t_evade():
    cands=sorted(glob.glob(f'{TC}/evade_*.csv'))
    if not cands: print('  [skip] evade'); return
    d=load(cands[0]); t=d['t']
    fig,ax=plt.subplots(2,1,figsize=(9,7),sharex=True)
    rm=d['red_missile']
    ax[0].plot(t,d['bank'],color=BLUE,lw=1.5,label='bank angle')
    ax[0].plot(t,d['g'],color=RED,lw=1.2,alpha=0.8,label='g-load')
    for i in range(1,len(t)):
        if rm[i]: ax[0].axvspan(t[i-1],t[i],color='red',alpha=0.06)
    ax[0].set_ylabel('bank (deg) / g'); ax[0].legend(fontsize=9)
    ax[0].set_title('Defensive evasion — maneuver while under fire (red shading = incoming missile)',fontweight='bold')
    ax[1].plot(t,d['alt_km'],color=GREEN,lw=1.5,label='altitude'); ax[1].plot(t,d['mach'],color=PURPLE,lw=1.5,label='Mach')
    ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('alt (km) / Mach'); ax[1].legend(fontsize=9)
    mdd=d.get('miss_distance_m'); md=mdd[~np.isnan(mdd)][-1] if mdd is not None and np.any(~np.isnan(mdd)) else None
    if md: ax[0].text(0.02,0.95,f'missile defeated — miss distance {md:,.0f} m',transform=ax[0].transAxes,
                      color=RED,fontweight='bold',va='top')
    fig.tight_layout(); fig.savefig(f'{OUT}/T3_evasion.png'); plt.close(fig); print('  T3_evasion.png')

# T4 ── g-loading (aircraft) + missile overload
def t_gload():
    p=f'{TC}/win_0.csv'; fp=f'{TC}/flyout_g.csv'
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    if os.path.exists(p):
        d=load(p); ax[0].plot(d['t'],d['g'],color=BLUE,lw=1.3)
        ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('load factor (g)'); ax[0].set_title('(a) Aircraft g-loading')
    if os.path.exists(fp):
        f=load(fp); tt=f['t']-f['t'][0]; ax[1].plot(tt,f['g'],color=RED,lw=1.5)
        ax[1].set_xlabel('time since launch (s)'); ax[1].set_ylabel('missile load factor (g)'); ax[1].set_title('(b) Missile overload')
    fig.suptitle('g-loading: aircraft & missile',fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{OUT}/T4_gloading.png'); plt.close(fig); print('  T4_gloading.png')

# T5 ── energy-maneuver signature (altitude vs Mach)
def t_energy_sig():
    p=f'{TC}/win_0.csv'
    if not os.path.exists(p): return
    d=load(p)
    fig,ax=plt.subplots(figsize=(7,5.2))
    sc=ax.scatter(d['mach'],d['alt_km'],c=d['t'],cmap='plasma',s=18)
    ax.plot(d['mach'],d['alt_km'],color='gray',lw=0.5,alpha=0.5)
    plt.colorbar(sc,label='time (s)')
    ax.set_xlabel('Mach'); ax.set_ylabel('altitude (km)')
    ax.set_title('Energy-maneuver signature\n(altitude–speed trajectory through the engagement)',fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{OUT}/T5_energy_signature.png'); plt.close(fig); print('  T5_energy_signature.png')

if __name__=='__main__':
    for fn in [t_phases,t_crank,t_evade,t_gload,t_energy_sig]:
        try: fn()
        except Exception as e: print(f'  [skip] {fn.__name__}: {e}')
    print(f'\ntactics figures in {OUT}')
