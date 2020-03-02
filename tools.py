from math import log,sqrt,atan2,pi,cos,sin

pow_law = lambda v2, z2, z_hat, a : v2 * ( (z_hat/z2) ** a) + 1e-06

def _wind_speed_to_vector(s, d):
    u = s * cos(pi/180 * (270-d))
    v = s * sin(pi/180 * (270-d))
    return u, v

def _pow_interpolation(v1, v2, z1, z2, z_hat):
    a1 = abs(v1)
    a2 = abs(v2)
    if v1 < 0 and v2 >= 0 or v1 >= 0 and v2 < 0:
        a2 = a1 + a2
    # elif s1 >= 0 and s2 < 0:
    #     a2 = a1 + a2

    a = log(a2/(a1+1e-06)+1e-06) / log(z2/z1)
    
    return round(pow_law(v2, z2, z_hat, a),3)


def wind_interp(s1, d1, s2, d2, z_hat):
    u1, v1 = _wind_speed_to_vector(s1, d1)
    u2, v2 = _wind_speed_to_vector(s2, d2)
    u_i = _pow_interpolation(u1, u2, 10, 100, z_hat)
    v_i = _pow_interpolation(v1, v2, 10, 100, z_hat)
    return u_i, v_i


def tmp_interp(t1, t2, z_hat):
    t_i = _pow_interpolation(t1, t2, 2, 100, z_hat)
    return t_i
