from jsb_gym.utils.autopilot import AircraftPIDAutopilot
from jsb_gym.simObjects.FDMObject import FDMObject
from jsb_gym.utils.navigation import delta_heading
import time

class F16BVR(FDMObject):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        self.aircraft_control = AircraftPIDAutopilot(self)
        self.aircraft_simulation_config = self.conf.aircraft_simulation
        


    def step(self, action):
        self.set_retract_gear()
        for _ in range(self.aircraft_simulation_config.Sim_time_step):
            
            self._step(action)


    def _step(self, action):

        # heading betwen 0-360 deg
        # altitude in meters
        # throttle between 0-1 
        action_heading = action[0]
        action_altitude = action[1]
        action_throttle = action[2]

        diff_head = delta_heading(action_heading, self.get_psi() )
        diff_alt = action_altitude - self.get_altitude()

        for _ in range(self.aircraft_simulation_config.Control_time_step):
            aileron_cmd, elevator_cmd, rudder_cmd	= self.aircraft_control.get_control_input(diff_head, diff_alt)
            self.command_aircraft(aileron_cmd, elevator_cmd, rudder_cmd, action_throttle)    
            self.fdm.run()
            if self.conf.fg_sleep_time is not None:
                time.sleep(self.conf.fg_sleep_time)


    def command_aircraft(self, aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd):

        self.set_aileron(aileron_cmd)
        self.set_elevator(elevator_cmd)
        self.set_rudder(rudder_cmd)
        # todo velocity control. for now set 0.49 throttle at the imput 
        self.set_throttle(throttle_cmd)

