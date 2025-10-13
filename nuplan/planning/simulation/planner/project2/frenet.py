import math
import numpy as np
class FrenetPath:
    def __init__(self):
        self.t = []
        # lateral movement in respect to arch s
        self.d = []
        self.d_d = []  # d'(s)
        self.d_dd = []  # d''(s)
        self.d_ddd = []  # d'''(t) in low speed / d'''(s) in high speed

        # longitudinal movement in respect to arch s
        self.s = []
        self.s_d = []  # s'(t)
        self.s_dd = []  # s''(t)
        self.s_ddd = []  # s'''(t)
    
        # final cost function
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        # longtitude speed in respect to chasis
        self.v = []
        self.a = []
        # curvature
        self.kappa = []
        # arch segment per step
        self.path_idx2s = []

def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
    """
    Convert state from Cartesian coordinate to Frenet coordinate

    Parameters
    ----------
        rs: reference line s-coordinate
        rx, ry: reference point coordinates
        rtheta: reference point heading
        rkappa: reference point curvature
        rdkappa: reference point curvature rate
        x, y: current position
        v_x: longtitude velocity
        a_x: longtitude acceleration
        theta: heading angle
        kappa: curvature

    Returns
    -------
        s_condition: [s(t), s'(t), s''(t)]
        d_condition: [d(s), d'(s), d''(s)]
    """
    dx = x - rx
    dy = y - ry

    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)

    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d = math.copysign(math.hypot(dx, dy), cross_rd_nd)

    delta_theta = theta - rtheta
    # Wrap delta_theta to [-pi, pi]
    delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))
    tan_delta_theta = math.tan(delta_theta)
    cos_delta_theta = math.cos(delta_theta)

    one_minus_kappa_r_d = 1 - rkappa * d
    d_dot = one_minus_kappa_r_d * tan_delta_theta

    kappa_r_d_prime = rdkappa * d + rkappa * d_dot

    d_ddot = (-kappa_r_d_prime * tan_delta_theta +
                one_minus_kappa_r_d / (cos_delta_theta * cos_delta_theta) *
                (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))

    s = rs
    s_dot = v * cos_delta_theta / one_minus_kappa_r_d

    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    s_ddot = (a * cos_delta_theta -
                s_dot * s_dot *
                (d_dot * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d

    return [s, s_dot, s_ddot], [d, d_dot, d_ddot]