from math import log,sqrt,atan2,pi,cos,sin


pow_law = lambda v10, v100, z_hat, a : v100 * ( (z_hat/100) ** a) + 1e-06

def _wind_speed_to_vector(s, d):
    u = s * cos(pi/180 * (270-d))
    v = s * sin(pi/180 * (270-d))
    return u, v

def _pow_interpolation(s1, s2, z):
    a1 = abs(s1)
    a2 = abs(s2)
    if s1 < 0 and s2 >= 0 or s1 >= 0 and s2 < 0:
        a2 = a1 + a2
    # elif s1 >= 0 and s2 < 0:
    #     a2 = a1 + a2

    a = log(a2/(a1+1e-06)+1e-06) / log(100/10)
    
    return round(pow_law(s1, s2, z, a),3)


def wind_interp(s1, d1, s2, d2, z):
    u1, v1 = _wind_speed_to_vector(s1, d1)
    u2, v2 = _wind_speed_to_vector(s2, d2)
    u_i = _pow_interpolation(u1, u2, z)
    v_i = _pow_interpolation(v1, v2, z)

    return u_i, v_i
