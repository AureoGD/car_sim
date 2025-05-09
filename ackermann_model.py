import numpy as np


class AckermannSlipModel:
    def __init__(
        self,
        L=0.21,
        b=0.137,
        r=0.05,
        L_f=None,
        L_r=None,
        slip_gain=0.1,
        use_differential=True,
    ):
        self.L = L
        self.b = b  # half track width
        self.r = r  # wheel radius
        self.slip_gain = slip_gain  # weight for slip ratio influence
        self.use_differential = use_differential

        # If not provided, assume equal split
        if L_f is None or L_r is None:
            self.L_f = self.L_r = L / 2.0
        else:
            self.L_f = L_f
            self.L_r = L_r

    def step(self, x, y, theta, phi_l, phi_r, delta, dt):
        v_l = self.r * phi_l
        v_r = self.r * phi_r
        v = (v_l + v_r) / 2.0

        if np.abs(np.tan(delta)) > 1e-4:
            R = self.L / np.tan(delta)
            v_l_expected = v * (R - self.b) / R
            v_r_expected = v * (R + self.b) / R
        else:
            R = np.inf
            v_l_expected = v_r_expected = v

        if self.use_differential:
            # Override v_l and v_r to match expected values (ideal Ackermann)
            v_l = v_l_expected
            v_r = v_r_expected
            slip_ratio = 0.0
        else:
            # Compute slip as mismatch in left/right wheel speeds
            slip_l = v_l - v_l_expected
            slip_r = v_r - v_r_expected
            slip_ratio = (slip_r - slip_l) / (2 * v) if v > 1e-3 else 0.0

        # Adjust effective steering angle due to slip
        delta_eff = delta + self.slip_gain * slip_ratio
        beta = np.arctan((self.L_r / (self.L_f + self.L_r)) * np.tan(delta_eff))

        # Update position
        x += v * np.cos(theta + beta) * dt
        y += v * np.sin(theta + beta) * dt
        theta += (v / self.L) * np.sin(beta) * dt

        return x, y, theta, beta
