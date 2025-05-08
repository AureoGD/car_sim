import numpy as np

class AckermannKinematics:
    def __init__(self, L=0.21, b=0.137, r_wheel=0.05):
        self.L = L
        self.b = b
        self.r_wheel = r_wheel

    def step(self, x, y, theta, rpm_l, rpm_r, delta, dt):
        v_l = (rpm_l * 2 * np.pi * self.r_wheel) / 60.0
        v_r = (rpm_r * 2 * np.pi * self.r_wheel) / 60.0

        v = (v_l + v_r) / 2.0
        theta_dot_wheels = (v_r - v_l) / (2.0 * self.b)
        theta_dot = (v / self.L) * np.tan(delta) + theta_dot_wheels

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += theta_dot * dt

        return x, y, theta