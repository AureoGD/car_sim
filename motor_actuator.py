class MotorActuator:
    def __init__(self, K_rpm_per_volt=100.0, tau=0.1):
        self.K = K_rpm_per_volt
        self.tau = tau
        self.rpm = 0.0
        self.voltage_cmd = 0.0

    def set_voltage(self, voltage):
        self.voltage_cmd = voltage

    def step(self, dt):
        rpm_target = self.K * self.voltage_cmd
        drpm = (rpm_target - self.rpm) / self.tau
        self.rpm += drpm * dt
        return self.rpm, rpm_target, self.voltage_cmd
