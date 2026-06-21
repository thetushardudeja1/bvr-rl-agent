import numpy as np
import pymap3d as pm
from jsb_gym.simObjects.FDMObject import FDMObject
from jsb_gym.utils.guidance_laws import PN
import time
from jsb_gym.utils.autopilot import MissilePIDAutopilot
from jsb_gym.utils.geospatial import dinstance_between_simObj_agent

class AAMBVR(FDMObject):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        self.missile_control = MissilePIDAutopilot(self)
        self.PN = PN(conf)
        self.missile_simulation_config = self.conf.missile_simulation
        self.missile_performance = self.conf.missile_performance
        self.reset_operational_state()

    def reset_operational_state(self):
        self.target = None

        self.target_lost = False
        self.target_hit = False
        self.active = False
        self.alive = True
        
        self.lost_count = 0
        self.notch_count = 0      # consecutive beaming samples (Doppler notch)
        self.target_norm = None
        self.target_norm_old = None
        # ballistic coast after lock loss (notch): the missile keeps flying dumb
        # instead of vanishing, until it goes subsonic or the coast cap expires.
        self.coasting = False
        self.coast_steps = 0


    def reset(self, lat, long, alt, vel, heading):
        super().reset(lat, long, alt, vel, heading)
        

    def is_alive(self):
        return self.alive

    def is_ready_to_launch(self):
        # NOTE: a coasting (lock-lost) missile is alive but must NOT be treated
        # as a fresh round — otherwise the carrier re-launches it and it
        # teleports back to the launcher mid-coast.
        return (not self.active and self.alive and not self.target_lost
                and not self.target_hit and not self.coasting)

    def is_active(self):
        return self.active and self.alive and not self.target_lost and not self.target_hit

    def is_target_hit(self):
        return not self.active and not self.alive and not self.target_lost and self.target_hit

    def is_traget_lost(self):
        return not self.active and not self.alive and self.target_lost and not self.target_hit


    def set_target(self, target):
        ''' 
        Set target aircraft object 
        Input: simObject
        '''
        self.target = target

    def launch(self, carrier_aircraft):
        if self.is_alive():
            self.active = True
            lat = carrier_aircraft.simObj.get_lat_gc_deg()
            long = carrier_aircraft.simObj.get_long_gc_deg()
            alt = carrier_aircraft.simObj.get_altitude()
            # BUG FIX: was get_mach() (~0.9) fed into a m/s slot — every
            # missile launched from a standstill. Inherit carrier airspeed.
            vel = carrier_aircraft.simObj.get_true_airspeed()
            heading = carrier_aircraft.simObj.get_psi()

            self.reset(lat, long, alt, vel, heading)


    def is_target_within_effective_range(self):

        if self.target_norm <= self.missile_performance.effective_radius:
            self.target_hit = True
            self.active = False
            self.alive = False

            # set hit to the target
            self.target.healthPoints -= 1
            return True
        return False

    def is_detonated_at_cpa(self):
        """Proximity / closest-point-of-approach fuze.

        Once armed (inside arm_range), the first sample where the distance stops
        shrinking and starts opening means the missile has just flown past its
        closest point. The true closest distance lies between the previous and
        current sample, so use min(target_norm, target_norm_old) as the miss
        distance. Detonate right there:
          - miss distance <= effective_radius  -> kill
          - otherwise                          -> clean miss (no orbit/spiral)
        Either way the missile dies at the pass, so it can never loop the target.
        """
        mp = self.missile_performance
        arm = getattr(mp, 'arm_range', 1500.0)
        if self.target_norm_old is None:
            return False
        if self.target_norm > arm:
            return False
        # still closing -> not yet at CPA
        if self.target_norm <= self.target_norm_old:
            return False

        cpa = min(self.target_norm, self.target_norm_old)
        self.active = False
        self.alive = False
        if cpa <= mp.effective_radius:
            self.target_hit = True
            self.target_lost = False
            self.target.healthPoints -= 1
        else:
            self.target_lost = True
            self.target_hit = False
        return True

    def is_mach_low(self):
        if self.missile_control.acceleration_stage_done() and self.get_mach() < self.missile_performance.target_lost_below_mach:
            self.target_lost = True
            self.target_hit = False
            self.active = False
            self.alive = False
            return True
        return False

    def is_alt_low(self):
        if self.get_altitude() < self.missile_performance.target_lost_below_alt:
            self.target_lost = True
            self.target_hit = False
            self.active = False
            self.alive = False
            return True
        return False

    def target_los_speed(self):
        """Target velocity component along the missile->target line-of-sight
        (m/s, absolute). ~0 means the target is flying perpendicular to the LOS
        (a beam maneuver); large means it is opening/closing along the LOS
        (drag or head-on). This is the Doppler the missile's radar sees."""
        m, t = self, self.target
        try:
            tlat, tlon, th = t.simObj.get_lat_gc_deg(), t.simObj.get_long_gc_deg(), t.simObj.get_altitude()
            tve, tvn, tvd  = t.simObj.get_v_east(), t.simObj.get_v_north(), t.simObj.get_v_down()
        except AttributeError:
            tlat, tlon, th = t.get_lat_gc_deg(), t.get_long_gc_deg(), t.get_altitude()
            tve, tvn, tvd  = t.get_v_east(), t.get_v_north(), t.get_v_down()
        e, n, u = pm.geodetic2enu(tlat, tlon, th,
                                  m.get_lat_gc_deg(), m.get_long_gc_deg(), m.get_altitude())
        norm = (e*e + n*n + u*u) ** 0.5
        if norm < 1.0:
            return 0.0
        return abs((e*tve + n*tvn + u*(-tvd)) / norm)

    def is_notched(self):
        """Doppler main-lobe-clutter notch. While inside the terminal seeker
        range, if the target's along-LOS speed stays below notch_vmin (it is in
        the beam) for notch_count consecutive samples, the seeker cannot separate
        it from ground clutter and drops lock. Beaming defeats the shot; dragging
        (high along-LOS speed) does not — so it cannot reinforce running away."""
        mp = self.missile_performance
        if not getattr(mp, 'notch_enable', False):
            return False
        if not self.missile_control.acceleration_stage_done():
            return False
        if self.target_norm is None or self.target_norm > mp.notch_range:
            self.notch_count = 0
            return False
            
        los_speed = self.target_los_speed()
        if los_speed < mp.notch_vmin:
            self.notch_count += 1
            if self.notch_count > mp.notch_count:
                # User Proof Logging
                print(f"\n[PROOF] 🎯 DOPPLER NOTCH SUCCESSFUL! Missile Radar Blinded.")
                print(f"  -> Relative Line-of-Sight Speed: {los_speed:.1f} m/s (Threshold: {mp.notch_vmin} m/s)")
                m_psi = self.get_psi()
                t_psi = self.target.simObj.get_psi()
                angle_diff = abs(m_psi - t_psi)
                if angle_diff > 180: angle_diff = 360 - angle_diff
                print(f"  -> Missile Heading: {m_psi:.1f}°")
                print(f"  -> Target Heading:  {t_psi:.1f}°")
                print(f"  -> Angle Difference: {angle_diff:.1f}° (90° = Perfect Perpendicular Notch)")

                # Lock lost — but the missile does NOT vanish. It stops guiding
                # and stops being a threat (active=False), yet stays alive and
                # coasts ballistically (see _coast) until it goes subsonic or the
                # coast cap expires. notch_count is left high so outcome
                # accounting still attributes the defeat to the notch.
                self.coasting = True
                self.coast_steps = 0
                self.active = False
                self.target_hit = False
                # freeze the missile's OWN current heading/altitude so the coast
                # demands no turn (otherwise the PID chases the stale guidance
                # heading at low speed, overshoots and loops back).
                self.coast_heading = self.get_psi()
                self.coast_alt = self.get_altitude()
                # target_lost stays False during the coast; set when it finally
                # dies, so it isn't double-counted as a guided "lost" shot.
                return True
        else:
            self.notch_count = 0
        return False

    def is_target_lost(self):
        """Missed-shot detection: after the boost phase, if the missile-target
        distance keeps OPENING for `lost_count` consecutive samples, the
        intercept geometry is gone — kill the missile. (Old version had the
        comparison inverted, reset a typo'd `count_lost`, and was disabled.)"""
        if not self.missile_control.acceleration_stage_done():
            return False
        if self.target_norm_old is not None and self.target_norm > self.target_norm_old:
            self.lost_count += 1
            if self.lost_count > self.missile_performance.lost_count:
                self.target_lost = True
                self.target_hit = False
                self.active = False
                self.alive = False
                return True
        else:
            self.lost_count = 0
        return False

    def step(self):

        if self.coasting:
            self._coast()
            return

        for _ in range(self.missile_simulation_config.Sim_time_step):
            self.target_norm = dinstance_between_simObj_agent(self, self.target)
            self.is_mach_low()
            self.is_alt_low()
            # direct sample-hit first (closing), then CPA fuze on the pass.
            # CPA runs before the slow opening-distance "lost" counter so the
            # missile detonates at the pass instead of orbiting for ~6 steps.
            if self.active:
                self.is_target_within_effective_range()
            if self.active:
                self.is_detonated_at_cpa()
            if self.active:
                self.is_target_lost()
            if self.active:
                self.is_notched()
            self.target_norm_old = self.target_norm

            if not self.active:
                break

            self._step()

    def _coast(self):
        """Ballistic flyout after the seeker lost lock (notch). The motor is out
        (throttle 0) and there is no guidance — the missile holds its last
        commanded heading/altitude and bleeds energy. It is removed when it goes
        subsonic (is_mach_low), hits the altitude floor, or the coast cap
        expires. This replaces the old 'vanish instantly on notch' behaviour so
        a defeated missile visibly sails past the target and falls away."""
        mp = self.missile_performance
        cap = getattr(mp, 'coast_max_substeps', 240)
        for _ in range(int(self.missile_simulation_config.Sim_time_step)):
            self.coast_steps += 1
            # keep a live distance reading for logging/Tacview
            self.target_norm = dinstance_between_simObj_agent(self, self.target)
            self.is_mach_low()
            self.is_alt_low()
            if not self.alive:
                self.coasting = False
                self.target_lost = True
                return
            if self.coast_steps > cap:
                self.coasting = False
                self.target_lost = True
                self.active = False
                self.alive = False
                return
            self._step_ballistic()

    def _step_ballistic(self):
        # hold the heading/altitude captured at the instant of lock loss -> the
        # autopilot only stabilises (wings level, no maneuver demand), so the
        # missile flies straight out and bleeds energy instead of looping back.
        heading_cmd = getattr(self, 'coast_heading', self.get_psi())
        altitude_cmd = getattr(self, 'coast_alt', self.get_altitude())
        for _ in range(int(self.missile_simulation_config.Control_time_step)):
            aileron_cmd, elevator_cmd, rudder_cmd, _ = \
                self.missile_control.get_control_input(heading_cmd, altitude_cmd)
            self.command_missile(aileron_cmd, elevator_cmd, rudder_cmd, 0.0)  # motor out
            self.fdm.run()
            if self.conf.fg_sleep_time is not None:
                time.sleep(self.conf.fg_sleep_time)

    def _step(self):

        # Freeze guidance at extremely close range (< 150m) to avoid PN singularity 
        # (division by near-zero distance causing infinite LOS rates and "revolving").
        if self.target_norm is not None and self.target_norm < 150.0:
            if hasattr(self, 'last_heading_cmd'):
                heading_cmd = self.last_heading_cmd
                altitude_cmd = self.last_altitude_cmd
            else:
                heading_cmd = self.get_psi()
                altitude_cmd = self.get_altitude()
        else:
            heading_cmd, altitude_cmd = self.PN.get_guidance(self)
            self.last_heading_cmd = heading_cmd
            self.last_altitude_cmd = altitude_cmd
        
        if self.target_norm > self.conf.missile_navigation.dive_at:
            altitude_cmd = self.conf.missile_navigation.alt_cruise
            self.conf.missile_navigation.tan_ref = self.conf.missile_navigation.tan_ref_max
        else:
            self.conf.missile_navigation.tan_ref = self.conf.missile_navigation.tan_ref_min
        
        for _ in range(self.missile_simulation_config.Control_time_step):
                
            aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd	= self.missile_control.get_control_input(heading_cmd, altitude_cmd)
            self.command_missile(aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd)  
            self.fdm.run()
            if self.conf.fg_sleep_time is not None:
                time.sleep(self.conf.fg_sleep_time)



    def command_missile(self, aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd):

        self.set_aileron(aileron_cmd)
        self.set_elevator(elevator_cmd)
        self.set_rudder(rudder_cmd)
        # todo velocity control. for now set 0.49 throttle at the imput 
        self.set_throttle(throttle_cmd)






