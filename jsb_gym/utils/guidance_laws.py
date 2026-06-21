import math
import pymap3d as pm
from jsb_gym.utils.geospatial import angle_between

class PN:
    def __init__(self, conf):
        self.N  = conf.missile_navigation.N
        self.dt = conf.missile_navigation.dt
        self.distance_to_target = None

    def get_target_ENU(self, missile):
        lat0 = missile.get_lat_gc_deg()
        lon0 = missile.get_long_gc_deg()
        h0   = missile.get_altitude()
        try:
            lat = missile.target.simObj.get_lat_gc_deg()
            lon = missile.target.simObj.get_long_gc_deg()
            h   = missile.target.simObj.get_altitude()
        except AttributeError:
            lat = missile.target.get_lat_gc_deg()
            lon = missile.target.get_long_gc_deg()
            h   = missile.target.get_altitude()
        e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        return (e, n, u)

    def get_target_v_ENU(self, missile):
        try:
            ve = missile.target.simObj.get_v_east()
            vn = missile.target.simObj.get_v_north()
            vu = -missile.target.simObj.get_v_down()
        except AttributeError:
            ve = missile.target.get_v_east()
            vn = missile.target.get_v_north()
            vu = -missile.target.get_v_down()
        return (ve, vn, vu)

    def get_v_ENU(self, missile):
        return (missile.get_v_east(), missile.get_v_north(), -missile.get_v_down())

    def get_target_altitude(self, missile):
        try:
            return missile.target.simObj.get_altitude()
        except AttributeError:
            return missile.target.get_altitude()

    def get_guidance(self, missile):
        # All arithmetic is scalar — no numpy allocations
        tex, tey, tez = self.get_target_ENU(missile)
        tvx, tvy, tvz = self.get_target_v_ENU(missile)
        vx,  vy,  vz  = self.get_v_ENU(missile)

        # v_rel = target_v - v
        rx, ry, rz = tvx-vx, tvy-vy, tvz-vz

        # rotation_vector = cross(target_ENU, v_rel) / dot(target_ENU, target_ENU)
        te_dot = tex*tex + tey*tey + tez*tez
        cx = tey*rz - tez*ry
        cy = tez*rx - tex*rz
        cz = tex*ry - tey*rx
        wx, wy, wz = cx/te_dot, cy/te_dot, cz/te_dot

        # acc_cmd = N * cross(v_rel, rotation_vector)
        ax = self.N * (ry*wz - rz*wy)
        ay = self.N * (rz*wx - rx*wz)
        az = self.N * (rx*wy - ry*wx)

        # v_PN = v + acc * dt  (zero z for heading calculation)
        v1 = (vx,  vy,  0.0)
        v2 = (vx + ax*self.dt, vy + ay*self.dt, 0.0)

        heading_PN = angle_between(v1, v2, in_deg=True)

        # heading_rel_direction: sign of cross(v1,v2)[2]
        cross_z = v1[0]*v2[1] - v1[1]*v2[0]
        hrd = 1 if cross_z < 0 else -1

        heading_cmd = (missile.get_psi() + hrd * heading_PN) % 360

        v_rel_norm = math.sqrt(rx*rx + ry*ry + rz*rz)
        te_norm    = math.sqrt(te_dot)
        time_to_impact = te_norm / v_rel_norm
        altitude_cmd   = self.get_target_altitude(missile) + tvz * time_to_impact

        self.distance_to_target = te_norm
        return heading_cmd, altitude_cmd