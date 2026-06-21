import jsbsim
import numpy as np
from jsb_gym.utils.units import m2f, f2m, lbs2kg

class FDMObject(object):
    def __init__(self, conf):
        
        self.fdm = jsbsim.FGFDMExec('.', None)
        self.fdm.set_debug_level(0)
        self.fdm.load_script(conf.fdm_xml)
        if conf.data_output_xml is not None:
            self.fdm.set_output_directive(conf.data_output_xml)

    def reset(self, lat, long, alt, vel, heading):
        
        # input  lat long in deg 
        self.fdm.set_property_value("ic/lat-gc-deg", lat)
        self.fdm.set_property_value("ic/long-gc-deg", long)
        # input  altitude in meters 
        self.fdm.set_property_value("ic/h-sl-ft", m2f(alt))
        # input vel in m/s      
        self.fdm['ic/u-fps'] = m2f(vel)
        # input  heading in deg    
        self.fdm['ic/psi-true-rad'] = np.radians(heading)

        self.fdm.set_property_value('propulsion/set-running', -1)

        self.fdm.reset_to_initial_conditions(0)
        
        self.fdm.run()
        
         
    def get_lat_gc_deg(self):
        return self.fdm['position/lat-gc-deg']

    def get_long_gc_deg(self):
        return self.fdm['position/long-gc-deg']

    def get_sim_time_sec(self):
        return self.fdm['simulation/sim-time-sec']

    def get_mach(self):
        return self.fdm['velocities/mach']


    def get_phi(self, in_deg = True):
        if in_deg:
            return  self.fdm['attitude/phi-deg']
        return self.fdm['attitude/phi-rad']

    def get_theta(self, in_deg = True):
        if in_deg:
            return self.fdm['attitude/theta-deg']
        return  self.fdm['attitude/theta-rad']

    def get_psi(self, in_deg = True):
        if in_deg:
            return  self.fdm['attitude/psi-deg']
        return  self.fdm['attitude/psi-rad']

    def get_altitude(self):
        return self.fdm['position/h-sl-meters']


    def get_true_airspeed(self):
        return f2m(self.fdm['velocities/vt-fps'])


    def get_v_north(self):
        return f2m(self.fdm['velocities/v-north-fps'])

    def get_v_east(self):
        # returns vector component for velocity in east direction in meters per second
        return f2m(self.fdm['velocities/v-east-fps'])

    def get_v_down(self):
        return f2m(self.fdm['velocities/v-down-fps'])


    def get_u(self):
        return f2m(self.fdm['u-fps'])
    
    def get_total_fuel(self):
        return lbs2kg(self.fdm['propulsion/total-fuel-lbs'])

    def set_retract_gear(self):
        self.fdm['gear/gear-pos-norm'] = 0


    def set_aileron(self, cmd):
        # cmd range -1 to 1 
        self.fdm['fcs/aileron-cmd-norm'] = cmd

    def set_elevator(self, cmd):
        # cmd range -1 to 1 
        self.fdm['fcs/elevator-cmd-norm'] = cmd

    def set_rudder(self, cmd):
        # cmd range -1 to 1 
        self.fdm['fcs/rudder-cmd-norm'] = cmd

    def set_throttle(self, cmd):
        self.fdm['fcs/throttle-cmd-norm'] = cmd


    def print_status(self):
        print(f"Time: {self.get_sim_time_sec():.1f} s, Lat: {self.get_lat_gc_deg():.4f} deg, Long: {self.get_long_gc_deg():.4f} deg, Alt: {self.get_altitude():.1f} m, Mach: {self.get_mach():.3f}, Heading: {self.get_psi():.1f} deg")
