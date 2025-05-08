import matplotlib.pyplot as plt
import numpy as np


class ResultPlotter:
    def __init__(self, path_x, path_y):
        self.path_x = path_x
        self.path_y = path_y

    def plot_trajectory(self, traj_x, traj_y, traj_theta):
        plt.figure(figsize=(8, 6))
        plt.plot(self.path_x, self.path_y, label='Reference Path')
        plt.plot(traj_x, traj_y, 'r--', label='Vehicle Trajectory')
        plt.quiver(traj_x[::25], traj_y[::25], np.cos(traj_theta[::25]), np.sin(traj_theta[::25]),
                   scale=50, width=0.0025, color='red', label='Heading')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Trajectory vs. Reference Path')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_steering(self, deltas, dt):
        times = np.arange(len(deltas)) * dt
        plt.figure(figsize=(8, 3))
        plt.plot(times, np.rad2deg(deltas), label='Steering Angle (deg)')
        plt.xlabel('Time (s)')
        plt.ylabel('Steering Angle (°)')
        plt.title('Steering Angle Over Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_velocity_rpm(self, velocities, rpms, dt):
        times = np.arange(len(velocities)) * dt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(times, velocities, color='blue', label='Velocity (m/s)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(times, rpms, color='orange', label='RPM')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('RPM')
        ax2.grid(True)
        ax2.legend()

        plt.suptitle('Velocity and RPM Over Time')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()