from  data_classes import missile_dataclass as MDC

fdm_xml= 'scripts/AIM_test.xml'
#fdm_xml= 'scripts/AIM2.xml'
data_output_xml= None
fg_sleep_time= None

# NOTE: only phi_min/phi_max (roll clamp in MissilePIDAutopilot) are actually
# read anywhere. The other fields are dead config kept for dataclass
# compatibility — in particular alt_min/alt_max do NOT constrain the missile
# (the loft cruises at 15km); real altitude floor is
# missile_performance.target_lost_below_alt.
missile_limits =MDC.MissileLimits(phi_min=-180.0,
               phi_max=180.0,
               theta_min=-90.0,
               theta_max=90.0,
               psi_min=0.0,
               psi_max=360.0,
               alt_min=3e3,
               alt_max=12e3,
               thr_min=0.0,
               thr_max=1.0)

PIDGains_roll  = MDC.PIDGains(P=  1, I= 0.0,  D=  4.0, Deriv=0.0, Integ=0.0, Integ_max=0.1, Integ_min=-0.1)
PIDGains_pitch = MDC.PIDGains(P=0.01, I= 0.0,  D=  3.0, Deriv=0.0, Integ=0.0, Integ_max=0.1, Integ_min=-0.1)
PIDGains_heading = MDC.PIDGains(P=1.8, I= 0.0,  D=  4.0, Deriv=0.0, Integ=0.0, Integ_max=0.1, Integ_min=-0.1)

missile_PID_Gains = MDC.MissilePIDGains(
    Roll=PIDGains_roll,
    Pitch=PIDGains_pitch,
    Heading=PIDGains_heading,
)


missile_navigation = MDC.MissileNavigation(
    N = 3.0,  # was 2.0 — real PN uses 3-5; N=2 was too easy to out-turn
    dt = 0.1,
    cp = 360,
    acceleration_stage_in_sec = 2,
    dive_at= 15e3,  
    tan_ref= 2e3,
    tan_ref_min= 2e3,
    tan_ref_max= 6e3,
    theta_min_cruise= -30.0,
    theta_max_cruise= 30.0,
    theta_min= -70.0,
    theta_max= 70.0,
    alt_cruise= 15e3
)


missile_simulation =MDC.MissileSimulation(
    Sim_time_step=6,
    Control_time_step=10
)

missile_performance = MDC.MissilePerformance(
    target_lost_below_mach= 1.0,
    target_lost_below_alt= 0.0,
    # consecutive opening-distance samples (checked ~12x/sim-second) before
    # the shot is declared missed: 12 = 1s of sustained opening
    lost_count=12,
    effective_radius=300,
    # Doppler-notch seeker: a sustained beam maneuver (target velocity ~perp to
    # the missile LOS) inside the terminal seeker range defeats the shot. This
    # is what makes beaming a real, distinct evasion from energy-dragging.
    notch_enable=True,
    notch_vmin=75.0,     # m/s of along-LOS target velocity below which it notches
    notch_count=20,      # consecutive samples (~1.6s) of sustained beaming to
                         # break lock — costs real commitment so beaming is a
                         # tool, not a free immortality (was 10 -> self-play
                         # stalemated as both sides became un-killable)
    notch_range=20e3,    # only inside terminal seeker range (active homing phase)
    notch_min_range=1.5e3,  # below this the warhead decides the merge; no notch
)

