import math

def enu_to_body(v_enu, agent):
    # Inline the full Rx @ Ry @ Rz rotation using scalar math only.
    # No numpy matrix allocation — ~10x faster for 3-vectors.
    roll  = math.radians(agent.simObj.get_phi())
    pitch = math.radians(agent.simObj.get_theta())
    yaw   = math.radians((agent.simObj.get_psi() - 90) % 360)

    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    # Combined rotation matrix R = Rx(roll) @ Ry(pitch) @ Rz(yaw) applied to v_enu
    ex, ey, ez = float(v_enu[0]), float(v_enu[1]), float(v_enu[2])

    # Rz(yaw) first:  [[cy,-sy,0],[sy,cy,0],[0,0,1]] @ v
    x1 =  cy*ex - sy*ey
    y1 =  sy*ex + cy*ey
    z1 = ez

    # Ry(pitch)
    x2 =  cp*x1 + sp*z1
    y2 = y1
    z2 = -sp*x1 + cp*z1

    # Rx(roll)
    x3 = x2
    y3 =  cr*y2 - sr*z2
    z3 =  sr*y2 + cr*z2

    return (x3, y3, z3)


if __name__ == "__main__":
    class _FakeAgent:
        class simObj:
            @staticmethod
            def get_phi(): return 0.0
            @staticmethod
            def get_theta(): return 0.0
            @staticmethod
            def get_psi(): return 0.0
    print(enu_to_body([0, 0, 1], _FakeAgent()))