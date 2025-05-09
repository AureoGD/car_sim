import numpy as np
from velocity_controller import VelocityPID
from base_velocity_controller import BaseVelocityController
from motor_actuator import MotorActuator
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
        v_ref=1.0,
        dt=0.01,
        T=20.0,
        use_velocity_pid=True,
        velocity_controller: BaseVelocityController = None,
        dt_traj=0.1,
        dt_ctrl=0.05,
        dt_act=0.02,
        dt_dyn=0.01,
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
            "v_cmd": [],
            "rpm_mean": [],
        }

        self.delta_cmd = 0.0
        self.v_cmd = v_ref
        self.rpm_l_cmd = 0.0
        self.rpm_r_cmd = 0.0

        self.motor_left = MotorActuator()
        self.motor_right = MotorActuator()

        if velocity_controller is not None:
            self.velocity_controller = velocity_controller
        else:
            self.velocity_controller = VelocityPID(kp=0.025, ki=0.01, kd=0.00001)

    def trajectory_control(self):
        v_meas = 0.0
        if len(self.rpms) > 0:
            rpm = self.rpms[-1]
            v_meas = (rpm * 2 * np.pi * self.model.r) / 60.0

        self.delta_cmd, _ = self.controller.compute_steering_angle(
            self.x, self.y, self.theta, v_meas, self.model.L
        )

    def velocity_control(self):
        v_meas = 0.0
        if len(self.rpms) > 0:
            rpm = self.rpms[-1]
            v_meas = (rpm * 2 * np.pi * self.model.r) / 60.0

        if self.use_velocity_pid:
            self.v_cmd = self.v_ref + self.velocity_controller.control(
                self.v_ref, v_meas, self.dt_ctrl
            )
        else:
            self.v_cmd = self.v_ref

        self.v_cmd = np.clip(self.v_cmd, 0, 3.0)

    def rpm_control(self):
        rpm_cmd = (self.v_cmd * 60) / (2 * np.pi * self.model.r)
        self.rpm_l_cmd = rpm_cmd
        self.rpm_r_cmd = rpm_cmd
        # PI do motor
        voltage_cmd = self.v_cmd / self.model.r / self.motor_left.K
        #
        self.motor_left.set_voltage(voltage_cmd)
        self.motor_right.set_voltage(voltage_cmd)
        self.system_log["voltage_l"].append(voltage_cmd)
        self.system_log["voltage_r"].append(voltage_cmd)
        self.system_log["rpm_l_ref"].append(self.rpm_l_cmd)
        self.system_log["rpm_r_ref"].append(self.rpm_r_cmd)

    def actuator_dynamics(self):
        self.rpm_l, _, _ = self.motor_left.step(self.dt_act)
        self.rpm_r, _, _ = self.motor_right.step(self.dt_act)
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
        self.system_log["delta_cmd"].append(self.delta_cmd)
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
        return {k: np.array(v) for k, v in self.system_log.items()}

    def visualize_with_pybullet(self, box_size=(0.4, 0.2, 0.1), sleep_time=0.01):
        import os
        import math
        import pybullet as p
        import pybullet_data

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Disable debug overlays
        for flag in [
            p.COV_ENABLE_GUI,
            p.COV_ENABLE_SHADOWS,
            p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        ]:
            p.configureDebugVisualizer(flag, 0)

        # White ground box dimensions
        width = 30
        height = 30
        thickness = 0.01

        # Create a simple white ground plane (box)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[width, height, thickness / 2],
            rgbaColor=[0.25, 0.25, 0.25, 0.8],  # White color
            specularColor=[0, 0, 0],
        )

        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=[width / 2, height / 2, thickness / 2]
        )

        ground_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[width / 2, height / 2, thickness / 2],
            baseOrientation=[0, 0, 0, 1],  # No rotation
        )

        # Create red box car
        car_shape_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[s / 2 for s in box_size],
            rgbaColor=[1, 0, 0, 1],
        )
        car_id = p.createMultiBody(
            baseMass=1,
            baseVisualShapeIndex=car_shape_id,
            basePosition=[self.traj_x[0], self.traj_y[0], box_size[2] / 2],
        )

        # Run simulation
        for x, y, theta in zip(self.traj_x, self.traj_y, self.traj_theta):
            orn = p.getQuaternionFromEuler([0, 0, theta])
            p.resetBasePositionAndOrientation(car_id, [x, y, box_size[2] / 2], orn)

            # p.resetDebugVisualizerCamera(
            #     cameraDistance=4.0,
            #     cameraYaw=0,
            #     cameraPitch=-30,
            #     cameraTargetPosition=[x, y, 0.5],
            # )

            p.stepSimulation()
            time.sleep(sleep_time)

        p.disconnect()
