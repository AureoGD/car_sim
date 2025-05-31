import numpy as np


class AckermannSlipModel:
    """
    A kinematic model for an Ackermann steering vehicle, with an option
    to simulate the effects of not having a mechanical differential.

    The model simulates how discrepancies in rear wheel speeds (when a
    differential is not used) can lead to an effective change in the
    steering angle, typically causing understeer.
    """

    def __init__(
            self,
            L=0.21,  # float: Wheelbase of the vehicle (meters)
            b=0.137,  # float: Half of the track width (distance from centerline to a wheel) (meters)
            r=0.05,  # float: Wheel radius (meters)
            L_f=None,  # float, optional: Distance from Center of Gravity (CoG) to front axle (meters)
            L_r=None,  # float, optional: Distance from Center of Gravity (CoG) to rear axle (meters)
            slip_gain=0.1,  # float: Gain factor determining how much wheel slip difference affects steering
            use_mechanical_differential=True,  # bool: True if an ideal differential is assumed, False otherwise
    ):
        """
        Initializes the Ackermann slip model.

        Args:
            L (float): Wheelbase of the vehicle (m).
            b (float): Half of the track width (m).
            r (float): Wheel radius (m).
            L_f (float, optional): Distance from CoG to front axle (m).
                                   If None, CoG is assumed to be at the midpoint of the wheelbase.
            L_r (float, optional): Distance from CoG to rear axle (m).
                                   If None, CoG is assumed to be at the midpoint of the wheelbase.
            slip_gain (float): A tuning parameter that scales the effect of the calculated
                               slip ratio on the effective steering angle. Higher values
                               mean slip has a more pronounced effect.
            use_mechanical_differential (bool): If True, the model assumes an ideal differential, and
                                     wheel speeds will match Ackermann conditions.
                                     If False, actual wheel speeds are used, and any deviation
                                     from Ackermann conditions can induce slip effects.
        """
        self.L = L
        self.b = b
        self.r = r
        self.slip_gain = slip_gain
        self.use_mechanical_differential = use_mechanical_differential

        # If distances to CoG from axles are not provided, assume CoG is at the midpoint.
        # L = L_f + L_r must hold.
        if L_f is None or L_r is None:
            self.L_f = self.L / 2.0
            self.L_r = self.L / 2.0
        else:
            self.L_f = L_f
            self.L_r = L_r
            if not np.isclose(self.L, self.L_f + self.L_r):
                raise ValueError("Wheelbase L must be equal to L_f + L_r.")

    def step(self, x, y, theta, phi_l, phi_r, delta, dt):
        """
        Performs one simulation step of the kinematic model.

        Args:
            x (float): Current x-coordinate of the vehicle's CoG (m).
            y (float): Current y-coordinate of the vehicle's CoG (m).
            theta (float): Current orientation (yaw angle) of the vehicle (radians).
            phi_l (float): Angular velocity of the left rear wheel (rad/s).
            phi_r (float): Angular velocity of the right rear wheel (rad/s).
            delta (float): Commanded steering angle of the front wheels (radians).
                           Positive for a left turn, negative for a right turn (if using SAE convention,
                           but often robotics uses positive for left turn / counter-clockwise yaw).
                           The model internally handles signs based on geometry.
            dt (float): Time step for the simulation (s).

        Returns:
            tuple: (x_new, y_new, theta_new, beta)
                x_new (float): Updated x-coordinate (m).
                y_new (float): Updated y-coordinate (m).
                theta_new (float): Updated orientation (radians).
                beta (float): Calculated vehicle sideslip angle at CoG (radians).
        """
        # Calculate linear speeds of left and right wheels
        v_l_actual = self.r * phi_l
        v_r_actual = self.r * phi_r

        # Calculate average longitudinal velocity of the vehicle (center of the rear axle)
        v = (v_l_actual + v_r_actual) / 2.0

        # Calculate expected wheel speeds for perfect Ackermann steering
        # R is the turning radius from the center of the rear axle to the turning point.
        # A very small delta (straight driving) implies a very large or infinite turning radius.
        if np.abs(np.tan(
                delta)) > 1e-6:  # Avoid division by zero for very small delta
            R_turn = self.L / np.tan(
                delta)  # Turning radius w.r.t. rear axle center
            # Expected speeds for inner and outer wheels to maintain Ackermann steering
            # v_l_expected is for the wheel on the inside of the turn
            # v_r_expected is for the wheel on the outside of the turn
            # Assuming delta > 0 is a left turn (R_turn > 0), left wheel is inner.
            # If delta < 0 is a right turn (R_turn < 0), right wheel is inner.
            # The formulas v * (R - b) / R and v * (R + b) / R handle this correctly
            # if 'b' is always positive half track width.
            # Example: Left turn (delta > 0, tan(delta) > 0). R_turn is positive.
            #   Left (inner) wheel: v * (R_turn - self.b) / R_turn
            #   Right (outer) wheel: v * (R_turn + self.b) / R_turn
            v_l_expected = v * (R_turn - self.b) / R_turn
            v_r_expected = v * (R_turn + self.b) / R_turn
        else:  # Driving straight or nearly straight
            R_turn = np.inf  # Effectively infinite turning radius
            v_l_expected = v
            v_r_expected = v

        slip_ratio_effect = 0.0
        if self.use_mechanical_differential:
            # If using an ideal differential, actual wheel speeds are assumed to match expected Ackermann speeds.
            # This means no slip is induced by the differential action itself.
            # (Note: This doesn't preclude tire slip due to other dynamic effects not modeled here)
            v_l = v_l_expected
            v_r = v_r_expected
            # slip_ratio_effect remains 0.0
        else:
            # No differential: use the actual commanded wheel speeds
            v_l = v_l_actual
            v_r = v_r_actual

            # Compute slip as the difference between actual speed and expected Ackermann speed for each wheel.
            # A positive slip_x means the wheel is moving faster than its Ackermann ideal.
            # A negative slip_x means the wheel is moving slower than its Ackermann ideal.
            slip_l = v_l - v_l_expected  # Slip of the left wheel
            slip_r = v_r - v_r_expected  # Slip of the right wheel

            # Calculate a slip ratio based on the difference in slips between right and left wheels.
            # This ratio aims to capture how the speed mismatch (due to no differential)
            # might induce a yawing moment or an effective change in steering.
            # If v is very small, avoid division by zero.
            if abs(v) > 1e-3:
                # (slip_r - slip_l) positive: right wheel slips "more forward" or left wheel "more backward".
                # Example: Right turn (delta < 0). R_turn is negative.
                #   v_l_expected (outer) > v_r_expected (inner).
                #   If v_l_actual = v_r_actual = v (e.g. motors commanded same speed):
                #     slip_l = v - v_l_expected (negative)
                #     slip_r = v - v_r_expected (positive)
                #     slip_r - slip_l is positive.
                #   This positive (slip_r - slip_l) will make delta_eff more positive (or less negative).
                #   If delta was negative (right turn), delta_eff becomes less negative,
                #   meaning a reduced effective right turn (understeer).
                slip_ratio_effect = (slip_r - slip_l) / (2 * v)
            else:
                slip_ratio_effect = 0.0

        # Adjust effective steering angle due to the slip effect
        # The slip_gain parameter scales the influence of the slip_ratio_effect.
        delta_eff = delta + self.slip_gain * slip_ratio_effect

        # Calculate vehicle sideslip angle (beta) at the CoG.
        # This is based on the kinematic bicycle model, using the effective steering angle.
        # beta = arctan( (L_r / L) * tan(delta_eff) )
        # where L_r is distance from CoG to rear axle, L is wheelbase.
        if np.isclose(self.L,
                      0):  # Avoid division by zero if wheelbase is zero
            beta = 0.0
        else:
            beta = np.arctan((self.L_r / self.L) * np.tan(delta_eff))

        # Update vehicle position and orientation using the kinematic bicycle model equations
        # x_dot = v * cos(theta + beta)
        # y_dot = v * sin(theta + beta)
        # theta_dot = (v / L_r) * sin(beta)  <-- Common formulation
        # OR theta_dot = (v * cos(beta) * tan(delta_eff)) / L <-- Another common formulation

        x_new = x + v * np.cos(theta + beta) * dt
        y_new = y + v * np.sin(theta + beta) * dt

        # Yaw rate calculation:
        # The formula used here is theta_dot = (v / L) * sin(beta).
        # A common alternative in kinematic bicycle models is theta_dot = (v / L_r) * sin(beta),
        # where L_r is the distance from CoG to the rear axle.
        # If L_r = L (CoG at front axle), this formula is consistent.
        # If CoG is at midpoint (L_r = L/2), the alternative would be (2*v/L)*sin(beta).
        # This choice affects the yaw dynamics. Ensure it matches the intended model convention.
        if np.isclose(self.L,
                      0):  # Avoid division by zero if wheelbase is zero
            theta_new = theta  # No change in theta if L is zero
        else:
            theta_new = theta + (v / self.L) * np.sin(beta) * dt
            # Example alternative:
            # if np.isclose(self.L_r, 0): theta_new = theta
            # else: theta_new = theta + (v / self.L_r) * np.sin(beta) * dt

        return x_new, y_new, theta_new, beta
