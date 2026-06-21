"""
action_utils.py
===============
Convert a BT agent's high-level action (heading deg, altitude m) into the
RL policy's normalized [-1, 1] action space — the inverse of
BaseEnv.from_nn2agent. Used to build the imitation (expert) dataset.
"""
from jsb_gym.utils.scale import scale_between


def _clip11(x):
    if x < -1.0: return -1.0
    if x >  1.0: return  1.0
    return x


def bt_action_to_nn(bt_heading, bt_altitude, limits, bt_launch=False):
    """
    bt_heading  : degrees (0..360)
    bt_altitude : meters
    limits      : aircraft_limits dataclass (psi_min/max, alt_min/max)
    bt_launch   : BT.launch_missile flag — the expert's trigger decision
    returns     : [heading, altitude, throttle, launch] normalized to [-1,1]
    """
    h = _clip11(scale_between(bt_heading,  limits.psi_min, limits.psi_max))
    a = _clip11(scale_between(bt_altitude, limits.alt_min, limits.alt_max))
    # BT always commands throttle 0.49 -> RL maps action[2] <= 0 to 0.49
    t = -1.0
    l = 1.0 if bt_launch else -1.0
    return [h, a, t, l]
