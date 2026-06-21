from dataclasses import dataclass

@dataclass
class MissileLimits:
    phi_min: float
    phi_max: float
    theta_min: float
    theta_max: float
    psi_min: float
    psi_max: float
    alt_min: float
    alt_max: float
    thr_min: float
    thr_max: float


@dataclass
class PIDGains:
    P: float
    I: float
    D: float
    Deriv: float
    Integ: float
    Integ_max: float
    Integ_min: float

@dataclass
class MissilePIDGains:
    Roll: PIDGains
    Pitch: PIDGains
    Heading: PIDGains


@dataclass
class MissileNavigation:
    N: float
    dt: float
    cp: float
    acceleration_stage_in_sec: float
    dive_at: float  
    tan_ref: float
    tan_ref_min: float
    tan_ref_max: float
    theta_min_cruise: float
    theta_max_cruise: float
    theta_min: float
    theta_max: float
    alt_cruise: float
    
@dataclass
class MissileSimulation:
    Sim_time_step: float
    Control_time_step: float

@dataclass
class MissilePerformance:
    target_lost_below_mach: float
    target_lost_below_alt: float
    lost_count: float
    effective_radius: float
    # --- Doppler main-lobe-clutter notch (defaults keep legacy configs valid) ---
    # If the TARGET's velocity projected on the missile line-of-sight stays below
    # notch_vmin (m/s) — i.e. the target is "in the beam" — for notch_count
    # consecutive samples while inside notch_range (m, the active-seeker terminal
    # gate), the seeker drops lock. notch_enable toggles the whole mechanic.
    notch_enable: bool = False
    notch_vmin: float = 75.0
    notch_count: int = 10
    notch_range: float = 20e3
    notch_min_range: float = 1.5e3   # below this the warhead decides; no notch
    # --- Proximity / closest-point-of-approach (CPA) fuze ---
    # Once inside arm_range the missile is "armed". The instant the
    # missile->target distance stops shrinking and starts opening, the missile
    # has just passed its closest point: it detonates THERE. If the closest
    # distance reached was <= effective_radius it is a kill, otherwise a clean
    # miss. This stops the missile from orbiting/spiralling the target after an
    # overshoot, and catches near-misses that the coarse time-step would skip
    # straight past the discrete effective_radius sample.
    arm_range: float = 1500.0