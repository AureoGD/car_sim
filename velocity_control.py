from motor_controller import BaseController


class VelocityController(BaseController):
    """
    A velocity controller that uses two PID controllers to control
    the left and right wheel speeds independently.

    This controller takes desired linear and angular velocities for the vehicle
    and translates them into individual wheel RPM setpoints. It then uses
    separate PID controllers for each wheel to generate  commands
    (e.g., voltage or PWM) to achieve those RPMs.
    """

    def __init__(self,
                 kp: float = 1.5,
                 ki: float = 0.5,
                 kd: float = 0.0,
                 N: float = 3.0,
                 ts: float = 0.05,
                 sat_u: float = 1.0,
                 sat_l: float = 0):
        """
        Initializes the VelocityController with parameters for wheel kinematics
        and for the underlying PID controllers.

        Args:
            kp_ (float): Proportional gain for the individual PID controllers.
            ki_ (float): Integral gain for the individual PID controllers.
            kd_ (float): Derivative gain for the individual PID controllers.
            N_ (float): Derivative filter coefficient for the PID controllers.
            ts_ (float): Sampling time for the PID controllers in seconds.
            sat_u_ (float): Upper saturation limit for the PID output ( command).
            sat_l_ (float): Lower saturation limit for the PID output ( command).
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.N = N

        self.sat_u = sat_u
        self.sat_l = sat_l

        self.ts = ts
        self.e_ant = 0.0
        self.ui_ant = 0.0
        self.ud_ant = 0.0
        self.y_ant = 0.0
        
        # Correct Integral term coefficient (This was already correct)
        self.kit = (self.ki * self.ts) / 2.0
        
        # Corrected Derivative term coefficients using Backward Euler
        # Assuming N is the filter cutoff frequency in rad/s
        if self.kd > 0 and self.N > 0 and self.ts > 0:
            tau_f = 1.0 / self.N
            self.kdt1 = tau_f / (self.ts + tau_f)
            self.kdt2 = self.kd / (self.ts + tau_f)
        else:
            self.kdt1 = 0.0
            self.kdt2 = 0.0
    def control(self, ref: float, y: float) -> float:
        """
        Computes the control signal based on the reference and current process variable.
        Includes anti-windup logic.
        """
        # --- 1. Calculate Error ---
        e = ref - y

        # --- 2. Calculate Proportional Term ---
        up = self.kp * e

        # --- 3. Calculate Derivative Term (on measurement y) ---
        # Note: self.kdt1 and self.kdt2 should be pre-calculated correctly in _init_
        if self.kdt1 > 0:
            ud = self.kdt1 * self.ud_ant - self.kdt2 * (y - self.y_ant)
        else:
            ud = 0.0
            
        # --- 4. Calculate pre-saturated control signal ---
        # We calculate the integral term ui momentarily without storing it
        ui_temp = self.ui_ant + self.kit * (e + self.e_ant)
        u_pre_sat = up + ui_temp + ud
        
        # --- 5. Apply Saturation ---
        u = u_pre_sat
        if u > self.sat_u:
            u = self.sat_u
        elif u < self.sat_l:
            u = self.sat_l
            
        # --- 6. Anti-Windup (Clamping / Back-Calculation) ---
        # If the output was saturated, recalculate the integral term to prevent windup.
        # This ensures the integrator state (self.ui_ant) doesn't grow while output is clamped.
        self.ui_ant = u - up - ud

        # --- 7. Update states for next iteration ---
        self.e_ant = e
        self.ud_ant = ud
        self.y_ant = y
        
        return u

    def reset(self):
        """
        Resets any internal states of the controller.

        This method SHOULD be implemented by child classes if they maintain
        internal states that need to be reset (e.g., integrator sum, previous error).
        The default implementation does nothing.
        """
        self.e_ant = 0.0
        self.ui_ant = 0.0
        self.ud_ant = 0.0
        self.y_ant = 0.0
