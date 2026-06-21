"""
agents.py
=========
BVR Gym agent classes.
Fixed version — corrected is_own_missile_active() check.

Bug fixed:
    - is_own_missile_active() now uses is_active() instead of .active
      This correctly accounts for missile alive/lost/hit states.
"""

from jsb_gym.simObjects.aircraft import F16BVR
from jsb_gym.simObjects.missiles import AAMBVR
import numpy as np
from jsb_gym.utils.geospatial import dinstance_between_agents, relative_bearing_between_agents
from jsb_gym.utils.wez import max_launch_range_m, no_escape_range_m


class BaseBVRAgent():
    def __init__(self, conf, env):
        self.conf = conf
        self.agent_parameters = conf.agent_parameters
        self.aircraft_simObj_conf = conf.aircraft_simObj_conf
        self.missile_simObj_conf  = conf.missile_simObj_conf

        self.simObj = F16BVR(self.aircraft_simObj_conf)
        self.reset_agent()
        self.reset_ammo()

        self.env = env
        self.healthPoints = 1.0
        self.target = None

    def reset_agent(self):
        self.simObj.reset(
            lat     = self.agent_parameters.lat,
            long    = self.agent_parameters.long,
            alt     = self.agent_parameters.alt,
            vel     = self.agent_parameters.vel,
            heading = self.agent_parameters.heading)

    def set_target(self, target_agent):
        self.target = target_agent

    def is_own_missile_active(self):
        # FIX: use is_active() instead of .active directly
        # is_active() checks: active AND alive AND not lost AND not hit
        return any(self.ammo[i].is_active() for i in self.ammo.keys())

    def get_active_missile(self):
        """The first currently-airborne missile this agent has launched, or
        None. Used by the env to expose incoming-threat geometry to the policy."""
        for i in self.ammo:
            if self.ammo[i].is_active():
                return self.ammo[i]
        return None

    def apply_action(self, action):
        self.simObj.step(action)
        self.apply_missile_action()

    def apply_missile_action(self):
        for i in self.ammo:
            # step guided missiles AND ones that lost lock but are still
            # coasting ballistically (so a notched missile flies on instead of
            # vanishing mid-air).
            if self.ammo[i].is_active() or getattr(self.ammo[i], 'coasting', False):
                self.ammo[i].step()

    def launch_missile(self):
        # No-escape-zone gate at the chokepoint so BT, RL and frozen opponents
        # all fire only high-Pk shots the target cannot outrun (measured Rne,
        # aspect/altitude/speed dependent) — not marginal max-range shots that
        # bleed energy and miss.
        # launch_range_scale (default 1.0) lets the env randomize red's firing
        # doctrine per episode: <1 = press in for a close, high-energy shot
        # (favours the drag defence); >1 = take a longer shot (more time/energy
        # for the missile, more notch-vulnerable). Diversifying it forces the
        # learner to develop more than one evasion answer.
        gate = no_escape_range_m(self, self.target) * getattr(self, 'launch_range_scale', 1.0)
        if dinstance_between_agents(self, self.target) > gate:
            return
        if not self.is_own_missile_active():
            for i in self.ammo:
                if self.ammo[i].is_ready_to_launch():
                    self.ammo[i].set_target(self.target)
                    self.ammo[i].launch(self)
                    break

    def reset_ammo(self):
        self.ammo = {}
        if self.agent_parameters.ammo > 0:
            for i in range(self.agent_parameters.ammo):
                self.ammo[str(i)] = AAMBVR(self.missile_simObj_conf)


class BTBVRAgent(BaseBVRAgent):

    def load_BT(self, BT_model):
        self.BT = BT_model(self)

    def apply_action(self):
        self.BT.tick()
        heading  = self.BT.heading
        altitude = self.BT.altitude
        throttle = 0.49
        if self.BT.launch_missile:
            self.launch_missile()
        action = np.array([heading, altitude, throttle])
        super().apply_action(action)


class RLBVRAgent(BaseBVRAgent):

    # range gating now lives in launch_missile() via the measured WEZ table;
    # this keeps only the seeker-pointing constraint
    RELATIVE_BEARING_MAX = 40     # degrees

    def launch_conditions_met(self):
        relative_bearing = relative_bearing_between_agents(self, self.target)
        return abs(relative_bearing) < self.RELATIVE_BEARING_MAX

    def apply_action(self, action):
        # 4th action dim = launch intent (the policy owns the trigger, like
        # H3E's weapon action). WEZ table + bearing remain hard constraints.
        # The old hardcoded auto-launch dumped the whole magazine at max
        # range — measured Pk ~0 against an evading target.
        if len(action) > 3:
            if action[3] > 0.0 and self.launch_conditions_met():
                self.launch_missile()
            super().apply_action(action[:3])
        else:
            # legacy 3-dim path (old models): auto-launch
            if self.launch_conditions_met():
                self.launch_missile()
            super().apply_action(action)


class BTWVRAgent(BaseBVRAgent):

    def load_BT(self, BT_model):
        self.BT = BT_model(self)

    def apply_action(self):
        self.BT.tick()
        heading  = self.BT.heading
        altitude = self.BT.altitude
        throttle = 0.49
        action   = np.array([heading, altitude, throttle])
        super().apply_action(action)


class RLWVRAgent(BaseBVRAgent):

    def apply_action(self, action):
        super().apply_action(action)