import math
from jsb_gym.utils.control import PID
from jsb_gym.utils.navigation import roll_circle_clip, delta_heading


def _clip(v, lo, hi):
	return lo if v < lo else hi if v > hi else v


class AircraftPIDAutopilot:
	"""
	Aircraft PID controller — continuous proportional-bank design.

	The previous design was bang-bang: any heading error above 10 deg slammed
	a fixed Roll_max (80 deg) bank with a hard-coded elevator of -0.9, then
	snap-rolled the other way after overshooting — the "corkscrew wobble".
	This version commands bank proportionally to heading error and always
	tracks pitch from altitude error, so control authority fades smoothly
	to zero as the error closes. Scalar `math` is used throughout: NumPy on
	scalars is ~10x slower and this runs 60x per agent step.
	"""

	# deg of bank commanded per deg of heading error (saturates at Roll_max)
	BANK_PER_HEADING_ERR = 2.5

	def __init__(self, aircraft):
		self.aircraft = aircraft
		self.aircraft_PID_Gains = self.aircraft.conf.aircraft_PID_Gains
		self.aircraft_navigation = self.aircraft.conf.aircraft_navigation
		self.aircraft_limits = self.aircraft.conf.aircraft_limits
		self.reset_controllers()

	def reset_controllers(self):

		self.roll_PID = PID(self.aircraft_PID_Gains.Roll)
		self.roll_sec_PID = PID(self.aircraft_PID_Gains.Roll_sec)
		self.pitch_PID = PID(self.aircraft_PID_Gains.Pitch)

		self.rudder_cmd = 0.0
		self.elevator_cmd = 0.0
		self.aileron_cmd = 0.0

	def set_roll_PID(self, roll_ref, secondary_pid=False):
		roll_ref = _clip(roll_ref, self.aircraft_limits.phi_min, self.aircraft_limits.phi_max)
		diff = roll_circle_clip(roll_ref - self.aircraft.get_phi())
		if secondary_pid:
			cmd = -self.roll_sec_PID.update(current_value=diff)
		else:
			cmd = -self.roll_PID.update(current_value=diff)
		self.aileron_cmd = _clip(cmd, -1.0, 1.0)

	def get_control_input(self, diff_head, diff_alt):
		# Bank proportional to heading error, saturating at Roll_max.
		roll_max = self.aircraft_navigation.Roll_max
		roll_ref = _clip(self.BANK_PER_HEADING_ERR * diff_head, -roll_max, roll_max)
		# fine-tracking PID near wings-level to avoid limit-cycling on small errors
		self.set_roll_PID(roll_ref, secondary_pid=(abs(diff_head) < 10.0))

		# Pitch always tracks the altitude error (no more open-loop elevator).
		theta_ref = math.degrees(math.atan2(diff_alt, self.aircraft_navigation.Tan_ref))
		theta_ref = _clip(theta_ref,
		                  self.aircraft_navigation.Dive_theta_max,
		                  self.aircraft_navigation.Climb_theta_max)
		# In a steep bank the vertical lift component shrinks by cos(phi);
		# add back-pressure so turns hold altitude instead of dropping the nose.
		phi = math.radians(self.aircraft.get_phi())
		cos_phi = math.cos(phi)
		if abs(cos_phi) > 0.2:
			theta_ref += min(10.0, (1.0 / abs(cos_phi) - 1.0) * 8.0)
		self.set_pitch_PID(theta_ref)

		return self.aileron_cmd, self.elevator_cmd, self.rudder_cmd

	def set_pitch_PID(self, theta_ref):
		theta_ref = _clip(theta_ref, self.aircraft_limits.theta_min, self.aircraft_limits.theta_max)
		diff = theta_ref - self.aircraft.get_theta()
		cmd = self.pitch_PID.update(current_value=diff)
		self.elevator_cmd = _clip(cmd, -1.0, 1.0)


class MissilePIDAutopilot:
	"""
	Missile PID controller class
	"""

	def __init__(self, missile):
		self.missile = missile
		self.missile_PID_Gains = self.missile.conf.missile_PID_Gains
		self.missile_limits = self.missile.conf.missile_limits
		self.missile_navigation = self.missile.conf.missile_navigation
		self.reset_controllers()
	
	def reset_controllers(self):
		
		self.roll_PID = PID(self.missile_PID_Gains.Roll)
		self.pitch_PID = PID(self.missile_PID_Gains.Pitch)
		self.heading_PID = PID(self.missile_PID_Gains.Heading)

		self.rudder_cmd = 0.0
		self.elevator_cmd = 0.0
		self.aileron_cmd = 0.0

	# bank-to-turn: deg of bank per deg of heading error (saturates at ROLL_MAX)
	BANK_PER_HEAD = 4.0
	ROLL_MAX      = 75.0

	def get_control_input(self, heading_cmd, altitude_cmd):
		if not self.acceleration_stage_done():
			heading_cmd = self.missile.get_psi()
			altitude_cmd = self.missile.get_altitude()

		# Skid-to-turn: maintain zero roll, steer with pitch and yaw
		self.aileron_cmd = self.set_roll_PID(roll_ref=0.0)
		self.elevator_cmd = self.set_altitude_PID(ref=altitude_cmd)
		self.rudder_cmd = self.set_heading_PID(head_ref=heading_cmd)
		self.throttle_cmd = self.set_throttle()

		return self.aileron_cmd, self.elevator_cmd, self.rudder_cmd, self.throttle_cmd


	def set_roll_PID(self, roll_ref):
		roll_ref = _clip(roll_ref, self.missile_limits.phi_min, self.missile_limits.phi_max)
		diff = roll_circle_clip(roll_ref - self.missile.get_phi())
		cmd = -self.roll_PID.update(current_value=diff)
		return _clip(cmd, -1.0, 1.0)

	def set_pitch_PID(self, theta_ref):
		diff = theta_ref - self.missile.get_theta()
		pitch_cmd = self.pitch_PID.update(current_value=diff)
		return -_clip(pitch_cmd, -1.0, 1.0)

	def set_heading_PID(self, head_ref):
		diff = delta_heading(head_ref, self.missile.get_psi())
		cmd = self.heading_PID.update(current_value=diff)
		return _clip(cmd, -1.0, 1.0)

	def set_throttle(self,cmd = 0.7):
		return cmd

	def set_altitude_PID(self, ref):
		diff_atl = ref - self.missile.get_altitude()
		theta_ref = math.degrees(math.atan2(diff_atl, self.missile_navigation.tan_ref))
		theta_ref = _clip(theta_ref, self.missile_navigation.theta_min, self.missile_navigation.theta_max)
		return self.set_pitch_PID(theta_ref)


	def acceleration_stage_done(self):
		if self.missile.get_sim_time_sec() < self.missile_navigation.acceleration_stage_in_sec:
			# still accelerating
			return False
		else:
			return True







