from math import exp


class MotorActuator:

    def __init__(self, ol_pole, ol_gain, ts):
        """
            Models a first-order system G(s) = k / (s + a) discretized using Zero-Order Hold (ZOH).

            Parameters:
                ol_gain (float): The open-loop gain 'k'.
                ol_pole (float): The open-loop pole 'a'. Must be non-zero.
                                 'a' is typically positive for a stable pole.
                ts (float): The sampling time (Ts). Must be positive.

            Continuous-Time Transfer Function:
                G(s) = k / (s + a)

            Discrete-Time Transfer Function (ZOH):
                If a != 0:
                G(z) = Y(z)/U(z) = [ (k/a) * (1 - e^(-a*ts)) * z^(-1) ] / [ 1 - e^(-a*ts) * z^(-1) ]
            
            This can be expressed using intermediate coefficients as used in this class:
                alpha_val = k/a
                beta_val = 1 - e^(-a*ts)
                gamma_val = e^(-a*ts)
            So, G(z) = (alpha_val * beta_val * z^(-1)) / (1 - gamma_val * z^(-1))

            The corresponding difference equation implemented is: 
                y[k] = gamma_val * y[k-1] + alpha_val * beta_val * u[k-1]
        """

        if ol_pole == 0:
            raise ValueError(
                "ol_pole (a) cannot be zero for this specific discretization. "
                "The case G(s)=k/s (integrator) requires a different formula.")
        if ts <= 0:
            raise ValueError("ts (sampling time) must be positive.")

        # Coefficients for the difference equation
        # y[k] = gamma * y[k-1] + (alpha * beta) * u[k-1]
        self.alpha = ol_gain / ol_pole
        self.beta = 1 - exp(-ol_pole * ts)
        self.gamma = exp(-ol_pole * ts)

        # State variables / previous values
        self.y_ant = 0.0  # Represents y[k-1], the output at the previous time step
        self.u_ant = 0.0  # Represents u[k-1], the input at the previous time step

        # Current input value, to be applied in the next step as u_ant
        # This variable 'self.u' will store the input u[k]
        self.u = 0.0

    def update_u(self, new_u):
        """
            Updates the current input value u[k] for the system.
            This input will be considered as u[k-1] in the *next* call to step().
        """
        self.u = new_u

    def step(self):
        """
            Calculates the system's output y[k] based on the previous state (y[k-1], u[k-1]).
            It then updates the system's state for the next iteration.

            The output y[k] is determined by:
            y[k] = gamma * y[k-1] + (alpha * beta) * u[k-1]

            After calculation, y[k] becomes y[k-1] for the next step,
            and the current input u[k] (set via update_u) becomes u[k-1] for the next step.
        """
        # Calculate current output y[k] using y[k-1] (self.y_ant) and u[k-1] (self.u_ant)
        current_y = self.gamma * self.y_ant + self.alpha * self.beta * self.u_ant

        # Update previous output for the next iteration
        self.y_ant = current_y

        # Update previous input for the next iteration:
        # The input self.u (which is u[k]) now becomes self.u_ant (u[k-1]) for the next step.
        self.u_ant = self.u

        return current_y
