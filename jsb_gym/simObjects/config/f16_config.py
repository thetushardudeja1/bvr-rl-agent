from data_classes import aircraft_dataclass as ADC 

fdm_xml= 'scripts/f16_test.xml'
data_output_xml= None
fg_sleep_time= None

aircraft_limits =ADC.AircraftLimits(phi_min=-180.0,
               phi_max=180.0,
               theta_min=-90.0,
               theta_max=90.0,
               psi_min=0.0,
               psi_max=360.0,
               alt_min=3e3,
               alt_max=12e3,
               thr_min=0.2,
               thr_max=0.69)


PIDGains_roll         = ADC.PIDGains(P=  0.01, I= 0.0,  D=  0.9, Deriv=0.0, Integ=0.0, Integ_max=0.2, Integ_min=-0.2)
PIDGains_roll_sec     = ADC.PIDGains(P=  0.2, I= 0.0,  D=  0.2, Deriv=0.0, Integ=0.0, Integ_max=1.0, Integ_min=-1.0)
PIDGains_pitch        = ADC.PIDGains(P=  0.3, I= 0.0, D=   1.0, Deriv=0.0, Integ=0.0, Integ_max=1.0, Integ_min=-1.0)
PIDGains_rudder_theta = ADC.PIDGains(P=  0.05, I= 0.0, D=   1.0, Deriv=0.0, Integ=0.0, Integ_max=10.0, Integ_min=-10.0)
PIDGains_rudder_psi   = ADC.PIDGains(P=  0.05, I= 0.0, D=   1.0, Deriv=0.0, Integ=0.0, Integ_max=1.0, Integ_min=-1.0)
PIDGains_elevator_psi = ADC.PIDGains(P=  0.1, I= 0.0, D=   4.0, Deriv=0.0, Integ=0.0, Integ_max=10.0, Integ_min=-10.0)



aircraft_PID_Gains = ADC.AircraftPIDGains(
    Roll=PIDGains_roll,
    Roll_sec=PIDGains_roll_sec,
    Pitch=PIDGains_pitch,
    Rudder_theta=PIDGains_rudder_theta,
    Rudder_psi=PIDGains_rudder_psi,
    Elevator_psi=PIDGains_elevator_psi
)



aircraft_navigation = ADC.AircraftNavigation(
    Head_act_space_min =10,
    Head_act_space_max =35,
    Tan_ref=2e3,
    Mach_min=0.1,
    Mach_max=1.2,
    V_down_min=-850,
    V_down_max=850,
    Roll_max=55,  # was 80 — fixed-bank slamming was the core of the corkscrew wobble
    Pitch_max=60,
    Pitch_min=-60,
    Alt_act_space_min = 1e3,
    Alt_act_space_max = 2e3,
    Climb_theta_max=45,
    Dive_theta_max=-45,
    Theta_act_space_min=10,
    Theta_act_space_max=30
)


aircraft_simulation =ADC.AircraftSimulation(
    Sim_time_step=6,
    Control_time_step=10
)
