import math
import pymap3d as pm

def to_360(angle):
    return (angle + 360) % 360

def dinstance_between_agents(own_agent, target_agent):
    lat0 = own_agent.simObj.get_lat_gc_deg()
    lon0 = own_agent.simObj.get_long_gc_deg()
    h0   = own_agent.simObj.get_altitude()
    lat  = target_agent.simObj.get_lat_gc_deg()
    lon  = target_agent.simObj.get_long_gc_deg()
    h    = target_agent.simObj.get_altitude()
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    return math.sqrt(e*e + n*n + u*u)

def bearing_between_agents(own_agent, target_agent):
    lat0 = own_agent.simObj.get_lat_gc_deg()
    lon0 = own_agent.simObj.get_long_gc_deg()
    h0   = own_agent.simObj.get_altitude()
    lat  = target_agent.simObj.get_lat_gc_deg()
    lon  = target_agent.simObj.get_long_gc_deg()
    h    = target_agent.simObj.get_altitude()
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    return math.degrees(math.atan2(e, n))

def relative_bearing_between_agents(own_agent, target_agent):
    bearing_to_enemy = bearing_between_agents(own_agent, target_agent)
    own_heading = own_agent.simObj.get_psi()
    return (bearing_to_enemy - own_heading + 180) % 360 - 180

def unit_vector(vector):
    ax, ay, az = float(vector[0]), float(vector[1]), float(vector[2])
    n = math.sqrt(ax*ax + ay*ay + az*az)
    return (ax/n, ay/n, az/n)

def angle_between(v1, v2, in_deg=False):
    ''' Return angle between 0 to 180 deg, or 0 to pi '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dot = v1_u[0]*v2_u[0] + v1_u[1]*v2_u[1] + v1_u[2]*v2_u[2]
    if dot >  1.0: dot =  1.0
    if dot < -1.0: dot = -1.0
    angle = math.acos(dot)
    return math.degrees(angle) if in_deg else angle

def dinstance_between_simObj_agent(own_simObj, target_agent):
    lat0 = own_simObj.get_lat_gc_deg()
    lon0 = own_simObj.get_long_gc_deg()
    h0   = own_simObj.get_altitude()
    try:
        lat = target_agent.simObj.get_lat_gc_deg()
        lon = target_agent.simObj.get_long_gc_deg()
        h   = target_agent.simObj.get_altitude()
    except AttributeError:
        lat = target_agent.get_lat_gc_deg()
        lon = target_agent.get_long_gc_deg()
        h   = target_agent.get_altitude()
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    return math.sqrt(e*e + n*n + u*u)

def enu_between_agents(own_agent, target_agent):
    lat0 = own_agent.simObj.get_lat_gc_deg()
    lon0 = own_agent.simObj.get_long_gc_deg()
    h0   = own_agent.simObj.get_altitude()
    lat  = target_agent.simObj.get_lat_gc_deg()
    lon  = target_agent.simObj.get_long_gc_deg()
    h    = target_agent.simObj.get_altitude()
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    return e, n, u