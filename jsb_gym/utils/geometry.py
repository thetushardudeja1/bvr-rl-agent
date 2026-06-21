import math

def angle_between_vectors(v1, v2):
    # v1, v2 are 3-element sequences — use scalar math, no numpy dispatch overhead
    ax, ay, az = float(v1[0]), float(v1[1]), float(v1[2])
    bx, by, bz = float(v2[0]), float(v2[1]), float(v2[2])

    dot    = ax*bx + ay*by + az*bz
    norm_a = math.sqrt(ax*ax + ay*ay + az*az)
    norm_b = math.sqrt(bx*bx + by*by + bz*bz)

    cos_theta = dot / (norm_a * norm_b)
    # clamp to [-1, 1] without numpy
    if cos_theta >  1.0: cos_theta =  1.0
    if cos_theta < -1.0: cos_theta = -1.0

    return math.acos(cos_theta)
