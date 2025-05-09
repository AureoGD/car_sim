import numpy as np
from velocity_controller import VelocityPID
from base_velocity_controller import BaseVelocityController


class Simulator:
    def __init__(
        self,
        model,
        controller,
        x0=0.0,
        y0=0.0,
        theta0=0.0,
        v_ref=2.0,
        dt=0.05,
        T=20.0,
        use_velocity_pid=True,
        velocity_controller: BaseVelocityController = None,
    ):
        self.model = model
        self.controller = controller
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.v_ref = v_ref
        self.dt = dt
        self.T = T
        self.use_velocity_pid = use_velocity_pid

        self.traj_x = [x0]
        self.traj_y = [y0]
        self.traj_theta = [theta0]
        self.deltas = []
        self.velocities = []
        self.rpms = []

        if velocity_controller is not None:
            self.velocity_controller = velocity_controller
        else:
            self.velocity_controller = VelocityPID(kp=0.025, ki=0.01, kd=0.00001)

    def run(self):
        steps = int(self.T / self.dt)
        for _ in range(steps):
            v_meas = 0.0
            if len(self.rpms) > 0:
                rpm = self.rpms[-1]
                v_meas = (rpm * 2 * np.pi * self.model.r) / 60.0

            delta, _ = self.controller.compute_steering_angle(
                self.x, self.y, self.theta, v_meas, self.model.L
            )

            if self.use_velocity_pid:
                v_cmd = self.v_ref + self.velocity_controller.control(
                    self.v_ref, v_meas, self.dt
                )
            else:
                v_cmd = self.v_ref

            v_cmd = np.clip(v_cmd, 0, 3.0)
            rpm_cmd = (v_cmd * 60) / (2 * np.pi * self.model.r)

            result = self.model.step(
                self.x, self.y, self.theta, rpm_cmd, rpm_cmd, delta, self.dt
            )

            if isinstance(result, tuple) and len(result) == 4:
                self.x, self.y, self.theta, _ = result
            else:
                self.x, self.y, self.theta = result

            self.traj_x.append(self.x)
            self.traj_y.append(self.y)
            self.traj_theta.append(self.theta)
            self.deltas.append(delta)
            self.rpms.append(rpm_cmd)
            self.velocities.append(v_cmd)

    def get_trajectory(self):
        return (
            np.array(self.traj_x),
            np.array(self.traj_y),
            np.array(self.traj_theta),
            np.array(self.deltas),
            np.array(self.rpms),
            np.array(self.velocities),
        )
