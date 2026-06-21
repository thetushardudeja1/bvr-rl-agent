import math
from jsb_gym.utils.units import f2m
from jsb_gym.utils.geospatial import dinstance_between_agents, enu_between_agents
from jsb_gym.utils.rotation import enu_to_body
from jsb_gym.utils.geometry import angle_between_vectors

def gunsnap(r_wez, max_range=f2m(3_000), min_range=f2m(500)):
    if r_wez > max_range:
        return 0.0
    elif r_wez >= min_range:
        return (max_range - r_wez) / (max_range - min_range)
    return 0.0

def angle_to_target_in_body_frame_deg(own_agent, target_agent):
    e, n, u        = enu_between_agents(own_agent, target_agent)
    v_enu_body     = enu_to_body((e, n, u), own_agent)
    aircraft_nose  = (1.0, 0.0, 0.0)
    angle_to_target = angle_between_vectors(aircraft_nose, v_enu_body)
    return math.degrees(angle_to_target)

def in_gun_wez(own_agent, target_agent, angle_limit=2, max_range=3_000, min_range=500):
    d   = dinstance_between_agents(own_agent, target_agent)
    ang = angle_to_target_in_body_frame_deg(own_agent, target_agent)
    if abs(ang) < angle_limit:
        return gunsnap(d, max_range=f2m(max_range), min_range=f2m(min_range))
    return 0


# ---------------------------------------------------------------------------
# Missile launch envelope — derived from the MEASURED WEZ
# (missile_envelope_sweep.py -> experiments/wez/envelope.json), replacing the
# old hard-coded `dist < 60km` box that ignored aspect and altitude entirely.
# Nearest-bucket lookup in (target aspect, shooter altitude); launch allowed
# inside MARGIN * measured max-hit-range. Falls back to the legacy 60km box
# if the envelope file is missing.
# ---------------------------------------------------------------------------
import os, json
from jsb_gym.utils.navigation import delta_heading
from jsb_gym.utils.geospatial import bearing_between_agents

ENVELOPE_PATH = os.path.join('experiments', 'wez', 'envelope.json')
MARGIN = 0.9
_ASPECTS = (0, 90, 180)
_ALTS_KM = (5, 8, 11)
_VELS    = (250, 330, 400)   # shooter speed buckets (m/s) — launch energy
_missile_table = None

def _load_missile_table():
    global _missile_table
    if _missile_table is None:
        try:
            with open(ENVELOPE_PATH) as f:
                _missile_table = json.load(f)['rmax_km']
        except (OSError, KeyError, ValueError):
            _missile_table = {}
    return _missile_table

def _nearest(value, options):
    return min(options, key=lambda o: abs(o - value))

def target_aspect_deg(shooter, target):
    """180 = head-on (target flying at the shooter), 0 = tail-chase."""
    brg_t2s = bearing_between_agents(target, shooter)
    tgt_head = target.simObj.get_psi()
    return 180.0 - abs(delta_heading(brg_t2s, tgt_head))

def max_launch_range_m(shooter, target):
    table = _load_missile_table()
    if not table:
        return 60e3   # legacy fallback when no measured envelope available
    asp = _nearest(target_aspect_deg(shooter, target), _ASPECTS)
    alt = _nearest(shooter.simObj.get_altitude() / 1e3, _ALTS_KM)
    vel = _nearest(shooter.simObj.get_true_airspeed(), _VELS)
    key = f"asp{asp}_alt{alt}km_v{vel}"
    if key in table:
        rmax_km = table[key]
    else:
        # older 2-axis envelope file (no speed dimension)
        rmax_km = table.get(f"asp{asp}_alt{alt}km", 0)
    return MARGIN * rmax_km * 1e3


# ---------------------------------------------------------------------------
# No-escape zone (Rne) — measured against an EVADING (dragging) target
# (missile_rne_sweep.py -> experiments/wez/rne.json). Firing inside Rne yields a
# high-Pk shot the target cannot outrun, so the missile arrives with energy
# instead of bleeding out on a marginal max-range shot. This is the launch gate.
# ---------------------------------------------------------------------------
RNE_PATH = os.path.join('experiments', 'wez', 'rne.json')
_rne_table = None

def _load_rne_table():
    global _rne_table
    if _rne_table is None:
        try:
            with open(RNE_PATH) as f:
                _rne_table = json.load(f)['rne_km']
        except (OSError, KeyError, ValueError):
            _rne_table = {}
    return _rne_table

def no_escape_range_m(shooter, target):
    """Launch gate: max range for a high-Pk (no-escape) shot. Falls back to
    0.6 * Rmax if the measured Rne table is unavailable."""
    table = _load_rne_table()
    asp = _nearest(target_aspect_deg(shooter, target), _ASPECTS)
    alt = _nearest(shooter.simObj.get_altitude() / 1e3, _ALTS_KM)
    vel = _nearest(shooter.simObj.get_true_airspeed(), _VELS)
    if table:
        rne_km = table.get(f"asp{asp}_alt{alt}km_v{vel}", 0)
        return rne_km * 1e3
    return 0.6 * max_launch_range_m(shooter, target) / MARGIN
