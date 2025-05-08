from base_velocity_controller import BaseVelocityController

class VelocityPID(BaseVelocityController):
    def __init__(self, kp=0.1, ki=0.01, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, v_ref, v_meas, dt):
        error = v_ref - v_meas
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
