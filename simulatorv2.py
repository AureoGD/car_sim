import numpy as np
from velocity_controller import VelocityPID
from base_velocity_controller import BaseController
from motor_actuator import MotorActuator
from motor_controller import MotorPID
import pybullet as p
import pybullet_data
import time


class Simulator:

    def __init__(
        self,
        model,
        controller,
        x0=0.0,
        y0=0.0,
        theta0=0.0,
        v_ref=1,
        dt=0.002,
        T=20.0,
        use_velocity_pid=True,
        velocity_controller: BaseController = None,
        dt_traj=0.1,
        dt_ctrl=0.05,
        dt_act=0.002,
        dt_dyn=0.01,
        path_x=None,
        path_y=None,
        path_theta=None,
    ):
        self.model = model
        self.controller = controller
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.v_ref = v_ref
        self.dt = dt
        self.T = T
        self.steps = int(T / dt)
        self.use_velocity_pid = use_velocity_pid

        self.dt_traj = dt_traj
        self.dt_ctrl = dt_ctrl
        self.dt_act = dt_act
        self.dt_dyn = dt_dyn

        self.last_traj_time = -dt_traj
        self.last_ctrl_time = -dt_ctrl
        self.last_act_time = -dt_act
        self.last_dyn_time = -dt_dyn

        self.time = 0.0

        self.traj_x = [x0]
        self.traj_y = [y0]
        self.traj_theta = [theta0]
        self.deltas = []
        self.velocities = []
        self.rpms = []

        self.system_log = {
            "rpm_l_actual": [],
            "rpm_r_actual": [],
            "rpm_l_ref": [],
            "rpm_r_ref": [],
            "voltage_l": [],
            "voltage_r": [],
            "traj_x": [x0],
            "traj_y": [y0],
            "traj_theta": [theta0],
            "delta_cmd": [],
            "vel_l_cmd": [],
            "vel_r_cmd": [],
            "v_cmd": [],
            "rpm_mean": [],
        }

        self.delta_cmd = 0.0
        self.v_cmd = v_ref
        self.rpm_l_cmd = 0.0
        self.rpm_r_cmd = 0.0

        self.motor_left = MotorActuator(dt=self.dt_act)
        self.motor_right = MotorActuator(dt=self.dt_act)
        self.motor_left_controller = MotorPID()
        self.motor_right_controller = MotorPID()
        self.velocity_controller = (velocity_controller if velocity_controller is not None else VelocityPID(
            kp=0.183222, ki=13.61028, kd=0.0001))

        self.path_x = path_x
        self.path_y = path_y
        self.path_theta = path_theta

    def update_mpc_reference(self):
        if self.path_x is None or self.path_y is None or self.path_theta is None:
            return

        dists = np.hypot(np.array(self.path_x) - self.x, np.array(self.path_y) - self.y)
        idx = np.argmin(dists)

        N = self.controller.N
        ref_slice = slice(idx, idx + N)

        # Criando vetor de velocidades constantes 1.0 m/s
        path_v = np.ones(len(self.path_x)) * 1.0

        # Extrai fatia da referência incluindo velocidade constante
        x_refs = np.stack([
            self.path_x[ref_slice],
            self.path_y[ref_slice],
            self.path_theta[ref_slice],
            path_v[ref_slice],
        ],
                          axis=1)

        if x_refs.shape[0] < N:
            last_row = x_refs[-1]
            x_refs = np.vstack([x_refs, np.tile(last_row, (N - x_refs.shape[0], 1))])

        self.controller.set_reference(x_refs)

    def trajectory_control(self):
        v_meas = 0.0
        if len(self.rpms) > 0:
            rpm = self.rpms[-1]
            v_meas = (rpm * 2 * np.pi * self.model.r) / 60.0

        if self.controller.control_name == "mpc":
            self.update_mpc_reference()

        if self.controller.control_name in ["stanley", "pure pursuit"]:
            self.delta_cmd, _ = self.controller.compute_steering_angle(self.x, self.y, self.theta, v_meas, self.model.L)
            self.vl_ref = self.v_ref
            self.vr_ref = self.v_ref

        elif self.controller.control_name == "mpc":
            self.delta_cmd, (self.vl_ref,
                             self.vr_ref) = self.controller.compute_steering_angle(self.x, self.y, self.theta, v_meas,
                                                                                   self.model.L)
            self.system_log["delta_cmd"].append(self.delta_cmd)
            self.system_log["vel_l_cmd"].append(self.vl_ref)
            self.system_log["vel_r_cmd"].append(self.vr_ref)

    def velocity_control(self):
        v_meas = 0.0
        if len(self.rpms) > 0:
            rpm = self.rpms[-1]
            v_meas = (rpm * 2 * np.pi * self.model.r) / 60.0

        if self.controller.control_name == "mpc":
            return
        else:
            v_ref = self.v_ref

        if self.use_velocity_pid:
            u_ctrl = self.velocity_controller.control(v_ref, v_meas, self.dt_ctrl)
        else:
            u_ctrl = v_ref / 3.0    # Normalize if not using PID

        u_ctrl = np.clip(u_ctrl, 0.0, 1.2)
        self.motor_left_input = u_ctrl
        self.motor_right_input = u_ctrl
        self.v_cmd = v_ref

    def rpm_control(self):
        if self.controller.control_name == "mpc":
            self.rpm_l_cmd = (self.vl_ref * 60) / (2 * np.pi * self.model.r)
            self.rpm_r_cmd = (self.vr_ref * 60) / (2 * np.pi * self.model.r)
        else:
            rpm_cmd = (self.v_cmd * 60) / (2 * np.pi * self.model.r)
            self.rpm_l_cmd = rpm_cmd
            self.rpm_r_cmd = rpm_cmd

        if len(self.rpms) <= 0:
            self.rpm_l = 0
            self.rpm_r = 0

        self.motor_right_input = self.motor_right_controller.control(rmp_ref=self.rpm_r_cmd, rpm=self.rpm_r) * 6 / 100
        self.motor_left_input = self.motor_left_controller.control(self.rpm_l_cmd, self.rpm_l) * 6 / 100
        self.motor_left.u = self.motor_left_input
        self.motor_right.u = self.motor_right_input

        self.system_log["rpm_l_ref"].append(self.rpm_l_cmd)
        self.system_log["rpm_r_ref"].append(self.rpm_r_cmd)
        self.system_log["voltage_l"].append(self.motor_left_input)
        self.system_log["voltage_r"].append(self.motor_right_input)

    def actuator_dynamics(self):
        self.rpm_l = self.motor_left.step()
        self.rpm_r = self.motor_right.step()

        #print(f'rpm = {self.rpm_l}')
        self.system_log["rpm_l_actual"].append(self.rpm_l)
        self.system_log["rpm_r_actual"].append(self.rpm_r)

    def car_dynamics(self):
        result = self.model.step(
            self.x,
            self.y,
            self.theta,
            self.rpm_l,
            self.rpm_r,
            self.delta_cmd,
            self.dt_dyn,
        )

        if isinstance(result, tuple) and len(result) == 4:
            self.x, self.y, self.theta, _ = result
        else:
            self.x, self.y, self.theta = result

        self.traj_x.append(self.x)
        self.traj_y.append(self.y)
        self.traj_theta.append(self.theta)
        self.deltas.append(self.delta_cmd)
        self.rpms.append((self.rpm_l + self.rpm_r) / 2.0)
        self.velocities.append(self.v_cmd)

        self.system_log["traj_x"].append(self.x)
        self.system_log["traj_y"].append(self.y)
        self.system_log["traj_theta"].append(self.theta)
        self.system_log["v_cmd"].append(self.v_cmd)
        self.system_log["rpm_mean"].append((self.rpm_l + self.rpm_r) / 2.0)

    def run(self):
        for _ in range(self.steps):
            if self.time - self.last_traj_time >= self.dt_traj:
                self.trajectory_control()
                self.last_traj_time = self.time

            if self.time - self.last_ctrl_time >= self.dt_ctrl:
                self.velocity_control()
                self.rpm_control()
                self.last_ctrl_time = self.time

            if self.time - self.last_act_time >= self.dt_act:
                self.actuator_dynamics()
                self.last_act_time = self.time

            if self.time - self.last_dyn_time >= self.dt_dyn:
                self.car_dynamics()
                self.last_dyn_time = self.time

            self.time += self.dt

    def get_system_states(self):
        for k, v in self.system_log.items():
            try:
                arr = np.array(v)
            except Exception as e:
                print(f"Erro ao converter '{k}': {e}")
                for i, item in enumerate(v):
                    print(
                        f"  item[{i}] = {item}, type: {type(item)}, shape: {np.shape(item) if hasattr(item, 'shape') else 'scalar'}"
                    )
        return {k: np.array(v) for k, v in self.system_log.items()}
