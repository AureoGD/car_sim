class BaseController:
    """
    A simple base class for controllers.
    
    This class is intended to define a common interface for controllers
    that take a reference (setpoint) and a current measurement to compute
    a control output, and may have resettable internal states.
    """

    def __init__(self):
        """
        Base constructor.
        Child classes can call super().__init__() if they wish to extend this,
        though this simple base __init__ currently does not perform extensive setup.
        It can be used to set common attributes if needed in the future.
        """
        # You could set a default name or other common properties here if desired, e.g.:
        # self.controller_type = "GenericSISOController"
        pass    # Keep it minimal for now as MotorPID doesn't call super().__init__()

    def control(self, ref: float, y: float) -> float:
        """
        Computes the control signal based on the reference and current process variable.

        This method MUST be implemented by child classes like MotorPID.

        Args:
            ref (float): The reference value (setpoint) for the controlled variable.
            y (float): The current measured value of the controlled variable.

        Returns:
            float: The computed control signal.
        
        Raises:
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError("The control() method must be implemented by child classes.")

    def reset(self):
        """
        Resets any internal states of the controller.

        This method SHOULD be implemented by child classes if they maintain
        internal states that need to be reset (e.g., integrator sum, previous error).
        The default implementation does nothing.
        """
        pass


class MotorPID(BaseController):
    """
    A discrete PID controller with anti-windup and a filtered derivative term.

    This controller implements the parallel form of the PID algorithm:
    u(t) = Kp * e(t) + Ki * integral(e(t) dt) + Kd * de(t)/dt

    Discretization:
    - The integral term is discretized using the Tustin (bilinear) transformation:
      Ui[k] = Ui[k-1] + (Ki * Ts/2) * (e[k] + e[k-1])
    - The derivative term is filtered to reduce noise sensitivity and acts on the
      negative of the process variable (-y) to prevent "derivative kick" when
      the setpoint changes. The implemented discrete form is:
      Ud[k] = beta_D * Ud[k-1] - gamma_D * (y[k] - y[k-1])
      This corresponds to a continuous-time derivative term approximately
      Kd * s / (1 + (Td_std/N)s), where Td_std = Kd/Kp. The discretization
      of the filter part is based on backward Euler.

    Anti-Windup:
    - Implemented using back-calculation (also known as clamping). When the
      computed total output exceeds the saturation limits, the integral term's
      stored state (Ui[k-1] for the next step) is adjusted to be consistent
      with the saturated output. This prevents the integral term from accumulating
      excessively (winding up) while the output is saturated.
    """

    def __init__(
            self,
            kp: float = 1.5,
            ki: float = 0.5,
            kd: float = 0.0,
            N: float = 3.0,    # Derivative filter coefficient
            ts: float = 0.05,    # Sampling time in seconds
            sat_u: float = 100.0,    # Upper saturation limit for output
            sat_l: float = 0.0):    # Lower saturation limit for output
        """
        Initializes the PID controller with specified gains and parameters.

        Args:
            kp (float): Proportional gain (Kp).
            ki (float): Integral gain (Ki).
            kd (float): Derivative gain (Kd).
            N (float): Derivative filter coefficient. This value tunes the trade-off
                       between derivative responsiveness and noise filtering.
                       Typical values range from 2 to 20. A higher N means
                       a faster filter pole and less filtering.
            ts (float): Sampling time (Ts) in seconds. Must be positive.
            sat_u (float): Upper saturation limit for the controller output.
            sat_l (float): Lower saturation limit for the controller output.
                           Must be less than sat_u.
        """
        if ts <= 0:
            raise ValueError("Sampling time (ts) must be positive.")
        if N <= 0:
            raise ValueError("Derivative filter coefficient (N) must be positive.")
        if sat_l >= sat_u:
            raise ValueError("Lower saturation limit (sat_l) must be less than "
                             "upper saturation limit (sat_u).")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.N = N
        self.ts = ts

        self.sat_u = sat_u
        self.sat_l = sat_l

        # --- Integral term coefficient ---
        # Based on Tustin's method for Ki * integral(e dt):
        # Ui[k] = Ui[k-1] + alpha_I * (e[k] + e[k-1])
        self.alpha_I = self.ki * self.ts / 2.0

        # --- Derivative term coefficients ---
        # Td_std is the standard derivative time constant: Kd / Kp.
        # The derivative term implemented is equivalent to Kd * s / (1 + (Td_std/N)s)
        # acting on -y.
        Td_std: float
        if self.kp == 0:
            if self.kd != 0.0:
                # If Kp is 0, Td_std = Kd/Kp is undefined.
                # The current formulation of gamma_D includes Kp. If Kp is 0,
                # gamma_D becomes 0, effectively disabling the derivative term.
                # This might be acceptable, or a different formulation for an
                # ID controller (no Kp) would be needed if kd should act independently.
                print(f"Warning: PID kd={self.kd} is non-zero but kp=0. "
                      "The derivative term's effective gain (via gamma_D) will be zero "
                      "with the current Td_std = kd/kp based formulation. "
                      "Consider using non-zero kp or a different derivative structure "
                      "if an ID controller with active derivative is intended.")
            Td_std = 0.0    # Disables derivative if kp=0 or if kd=0
        else:    # kp != 0
            Td_std = self.kd / self.kp

        if Td_std == 0.0:    # Handles kd=0 or the kp=0 case above
            self.beta_D = 0.0    # No derivative history if Td_std is 0
            self.gamma_D = 0.0    # No derivative action if Td_std is 0
        else:
            # Denominator for beta_D and gamma_D calculation
            # This is Td_std + N*Ts. Should be positive if Td_std >= 0, N > 0, Ts > 0.
            denominator_D = Td_std + self.N * self.ts
            if denominator_D <= 0:    # Should not happen with valid parameters
                # (e.g. if Td_std became unexpectedly negative and large)
                print(f"Warning: Derivative denominator is non-positive ({denominator_D}). "
                      "Derivative term will be unstable or disabled.")
                self.beta_D = 1.0    # Freeze derivative term to avoid issues
                self.gamma_D = 0.0    # Disable derivative action
            else:
                self.beta_D = Td_std / denominator_D
                # Original gamma was (kp * N * Td_std) / denominator_D
                # This makes the derivative action Kp * (filtered Td_std * de/dt)
                # which is Kd * (filtered de/dt)
                self.gamma_D = (self.kp * self.N * Td_std) / denominator_D

        # Initialize states for the controller
        self.e_ant = 0.0    # Previous error e[k-1]
        self.ui_ant = 0.0    # Previous integral component Ui[k-1]
        self.ud_ant = 0.0    # Previous derivative component Ud[k-1]
        self.y_ant = 0.0    # Previous process variable y[k-1]

    def control(self, ref: float, y: float) -> float:
        """
        Computes the control signal based on the reference and current process variable.

        Args:
            ref (float): The reference value (setpoint).
            y (float): The current process variable (measurement).

        Returns:
            float: The computed control signal, clamped within saturation limits.
        """
        # Calculate current error
        e = ref - y

        # --- Proportional Term ---
        up = self.kp * e

        # --- Derivative Term ---
        # Acts on the change in process variable (-y) to avoid derivative kick.
        # Ud[k] = beta_D * Ud[k-1] - gamma_D * (y[k] - y[k-1])
        delta_y = y - self.y_ant
        ud = self.beta_D * self.ud_ant - self.gamma_D * delta_y

        # --- Integral Term (Tustin's method) ---
        # Calculate the candidate integral component for the current step.
        # Ui_candidate = Ui[k-1] + alpha_I * (e[k] + e[k-1])
        ui_candidate = self.ui_ant + self.alpha_I * (e + self.e_ant)

        # --- Tentative control output (before saturation) ---
        output_presat = up + ui_candidate + ud

        # --- Apply saturation to the total output ---
        output_saturated = output_presat
        if output_saturated > self.sat_u:
            output_saturated = self.sat_u
        elif output_saturated < self.sat_l:
            output_saturated = self.sat_l

        # --- Anti-Windup: Back-calculation for the integral term ---
        # Adjust the stored integral component (self.ui_ant for the next step)
        # so that it reflects the integral part that was actually achievable
        # given the saturation. If the output was saturated, self.ui_ant is
        # set to (saturated_output - proportional_part - derivative_part).
        # This prevents the integral state from accumulating (winding up)
        # when the controller output is limited.
        self.ui_ant = output_saturated - up - ud

        # --- Update previous values for the next iteration ---
        self.e_ant = e
        self.y_ant = y
        self.ud_ant = ud    # Store the calculated derivative component Ud[k]

        return output_saturated

    def reset(self):
        """
        Resets the internal states of the PID controller.
        This is useful when re-starting control, after a mode change,
        or to clear accumulated integral and derivative states.
        """
        self.e_ant = 0.0
        self.ui_ant = 0.0
        self.ud_ant = 0.0
        self.y_ant = 0.0
