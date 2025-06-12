import numpy as np
# from velocity_controller import VelocityPID
# from base_velocity_controller import BaseController
from motor_actuator import MotorActuator
from motor_controller import MotorPID
import pybullet as p
import pybullet_data
import time
from math import pi


class Simulator:

    def __init__(
        self,
        model,
        controller,
        use_velocity_controller=False,
        x0=0.0,
        y0=0.0,
        theta0=0.0,
        v_car_ref=1,
        T=20.0,
        dt=0.001,
        dt_path_controller=0.1,  # e.g., Path controller runs at 10Hz
        dt_speed_controller=0.05,  # e.g., Car speed loop at 20Hz
        dt_motor_controller=0.01,  # e.g., Motor RPM PIDs at 100Hz
        dt_motor_dynamics=0.002,  # e.g., Motor model updates at 500Hz (same as base tick)
        dt_vehicle_dynamics=0.005  # e.g., Main car model at 200Hz
    ):
        self.model = model
        self.controller = controller
        self.use_velocity_controller = use_velocity_controller
        self.x = x0
        self.y = y0
        self.theta = theta0

        # step time values
        self.time = 0.0
        self.dt = dt
        self.dt_path_controller = dt_path_controller
        self.dt_speed_controller = dt_speed_controller
        self.dt_motor_controller = dt_motor_controller
        self.dt_motor_dynamics = dt_motor_dynamics
        self.dt_vehicle_dynamics = dt_vehicle_dynamics

        self.T = T

        self.v_car_ref = v_car_ref
        self.car_speed = 0

        # Actuators objects and constants
        ol_pole = 11.58
        ol_gain = 42.74
        self.motor_l = MotorActuator(ol_pole, ol_gain, self.dt_motor_dynamics)
        self.motor_r = MotorActuator(ol_pole, ol_gain, self.dt_motor_dynamics)
        self.motor_l_controller = MotorPID(kp=0.4679,
                                           ki=5.4182,
                                           kd=0,
                                           ts=self.dt_motor_controller)
        self.motor_r_controller = MotorPID(kp=0.4679,
                                           ki=5.4182,
                                           kd=0,
                                           ts=self.dt_motor_controller)

        # velocity control for the car
        if self.use_velocity_controller:
            pass  # not implemented yet
        else:
            self.velocity_controller = None

        # Varibles for the motor control loop
        self.motor_cmd_l = 0
        self.motor_cmd_r = 0
        self.rpm_l_ref = 0
        self.rpm_r_ref = 0
        self.rpm_l = 0
        self.rpm_r = 0

        self.delta_cmd = 0
        self.vl_ref = 0
        self.vr_ref = 0
        self.delta_cmd = controller.current_delta
        # Logs

        self.system_log = {
            "rpm_ref": [],  # Store entries like {t, rpm_l_ref, rpm_r_ref}
            "rpms": [],  # Store entries like {t, rpm_l, rpm_r}
            "vehicle_pose": [],  # Store entries like {t, x, y, theta, beta}
            "vehicle_speeds":
            [],  # Store linear and angular mean velocity {t, vl, dtheta, vl_speed, vr_speed} 
            "control_delta": [],  # Store entries like {t, delta_cmd}
            "motor_cmd":
            [],  # Store the control action for each motor {t, motor_cmd_l, motor_cmd_r}
            "vel_cmd":
            [],  # Store the commanded velocitires {t, vl_ref, vr_ref}
            "mpc_du": [],  #Store {t, dvl, dvr, d_delta}
        }

        self.steps = int(self.T / self.dt)

        self.last_path_control_time = 0
        self.last_car_speed_control_time = 0
        self.last_motor_control_time = 0
        self.last_actuator_time = 0
        self.last_vehicle_dynamics_time = 0

    def run(self):
        for _ in range(self.steps):

            if self.time - self.last_path_control_time >= self.dt_path_controller:
                self._path_control()
                self.last_path_control_time = self.time

            if self.time - self.last_car_speed_control_time >= self.dt_speed_controller:
                self._car_speed_control()
                self.last_car_speed_control_time = self.time

            if self.time - self.last_motor_control_time >= self.dt_motor_controller:
                self._motor_control()
                self.last_motor_control_time = self.time

            if self.time - self.last_actuator_time >= self.dt_motor_dynamics:
                self._motor_dynamic()
                self.last_actuator_time = self.time

            if self.time - self.last_vehicle_dynamics_time >= self.dt_vehicle_dynamics:
                self._car_dynamics()
                self.last_vehicle_dynamics_time = self.time

            self.time += self.dt

    def _path_control(self):
        current_car_speed, vl_current_speed, vr_current_speed = self._car_speed(
        )
        self.controller.update_states(self.x, self.y, self.theta,
                                      current_car_speed, vl_current_speed,
                                      vr_current_speed, self.delta_cmd)

        control_commands = self.controller.compute_control()

        self.delta_cmd = control_commands.get('delta', 0.0)
        self.system_log["control_delta"].append([self.time, self.delta_cmd])

        if self.controller.control_name == "mpc":
            self.system_log["mpc_du"].append([
                self.time, self.controller.du[0][0], self.controller.du[1][0],
                self.controller.du[2][0]
            ])
            self.vl_ref = control_commands.get('v_left', 0.0)
            self.vr_ref = control_commands.get('v_right', 0.0)

            vl_ref_rads = self.vl_ref / self.model.r
            vr_ref_rads = self.vr_ref / self.model.r

            self.rpm_l_ref = self._rads_to_rpm(vl_ref_rads)
            self.rpm_r_ref = self._rads_to_rpm(vr_ref_rads)
            self.system_log["vel_cmd"].append(
                [self.time, self.vl_ref, self.vr_ref])
            self.system_log["rpm_ref"].append(
                [self.time, self.rpm_l_ref, self.rpm_r_ref])

    def _car_speed_control(self):
        if self.controller.control_name == "mpc":
            return

        if self.use_velocity_controller:
            car_speed, _, _ = self._car_speed()
            # add the control method in the future
            # cdm_vel_car = self.velocity_controller.control(self.v_car_ref, self.car_speed)    # return the var velocity in m/s
        else:
            cdm_vel_car = self.v_car_ref

        phi_cmd = cdm_vel_car / self.model.r
        rpm_cmd = self._rads_to_rpm(phi_cmd)
        self.rpm_l_ref = rpm_cmd
        self.rpm_r_ref = rpm_cmd
        self.system_log["vel_cmd"].append(
            [self.time, cdm_vel_car, cdm_vel_car])
        self.system_log["rpm_ref"].append(
            [self.time, self.rpm_l_ref, self.rpm_r_ref])

    def _motor_control(self):
        self.motor_cmd_l = self.motor_l_controller.control(
            self.rpm_l_ref, self.rpm_l)
        self.motor_cmd_r = self.motor_r_controller.control(
            self.rpm_r_ref, self.rpm_r)
        self.motor_l.update_u(self.motor_cmd_l)
        self.motor_r.update_u(self.motor_cmd_r)
        # Log the motor control output
        self.system_log["motor_cmd"].append(
            [self.time, self.motor_cmd_l, self.motor_cmd_r])

    def _motor_dynamic(self):
        self.rpm_l = self.motor_l.step()
        self.rpm_r = self.motor_r.step()
        # Log the motor rpm
        self.system_log["rpms"].append([self.time, self.rpm_l, self.rpm_r])

    def _car_dynamics(self):
        phi_l = self._rpm_to_rads(self.rpm_l)
        phi_r = self._rpm_to_rads(self.rpm_r)

        # self.theta currently holds theta_k-1 (theta from previous vehicle dynamics step)
        # result will contain the new state including new_theta (theta_k)
        result = self.model.step(self.x, self.y, self.theta, phi_l, phi_r,
                                 self.delta_cmd, self.dt_vehicle_dynamics)

        # Unpack new state
        new_x = result[0]
        new_y = result[1]
        new_theta = result[2]  # This is theta_k
        beta = result[3]  # Vehicle sideslip angle

        # Calculate angular velocity
        # omega = (theta_k - theta_k-1) / dt
        angular_velocity_rad_s = (new_theta -
                                  self.theta) / self.dt_vehicle_dynamics

        # Update vehicle state AFTER calculating angular velocity using the previous self.theta
        self.x = new_x
        self.y = new_y
        self.theta = new_theta  # Now self.theta is updated to theta_k

        # Log pose and speeds (now including angular velocity)
        # The key "vehicle_speeds" in system_log will now store [time, linear_v, angular_omega]
        self.system_log["vehicle_pose"].append(
            [self.time, self.x, self.y, self.theta, beta])
        car_speed, vl_speed, vr_speed = self._car_speed()
        self.system_log["vehicle_speeds"].append(
            [self.time, car_speed, angular_velocity_rad_s, vl_speed, vr_speed])

    # Auxiliary functions

    def _car_speed(self):
        phi_l = self._rpm_to_rads(self.rpm_l)
        phi_r = self._rpm_to_rads(self.rpm_r)
        v_l_linear = self.model.r * phi_l
        v_r_linear = self.model.r * phi_r
        car_speed = (v_l_linear + v_r_linear) / 2.0
        return car_speed, v_l_linear, v_r_linear

    def _rpm_to_rads(self, rpm):
        return rpm * (2 * pi) / 60.0

    def _rads_to_rpm(self, rads):
        return rads * 60.0 / (2 * pi)
