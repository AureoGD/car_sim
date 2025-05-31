import matplotlib.pyplot as plt
import numpy as np
import pandas as pd    # Pandas can be very helpful for handling log data


class SimulationPlotter:

    def __init__(self, log_data_list: list, labels: list):
        """
        Initializes the SimulationPlotter with a list of log data dictionaries
        and corresponding labels for an easy comparison.

        Args:
            log_data_list (list): A list of system_log dictionaries from the Simulator.
                                  Each dictionary contains the logged data from one simulation run.
            labels (list): A list of strings corresponding to each log_data dictionary,
                           to be used in plot legends (e.g., ["With Diff", "Without Diff"]).
        """
        if len(log_data_list) != len(labels):
            raise ValueError("The number of log_data sets must match the number of labels.")

        self.log_data_list = log_data_list
        self.labels = labels

        # Pre-process logs into pandas DataFrames for easier handling (optional but recommended)
        self.processed_logs = []
        for log_data in self.log_data_list:
            processed = {}
            for key, data_entries in log_data.items():
                if not data_entries:    # Skip empty logs
                    processed[key] = pd.DataFrame()    # Empty DataFrame
                    continue

                # Determine columns based on key or data structure
                if key == "rpm_ref" or key == "rpms" or key == "motor_cmd":
                    columns = ['time', 'left', 'right']
                elif key == "vehicle_pose":
                    columns = ['time', 'x', 'y', 'theta', 'beta']
                elif key == "vehicle_speeds":
                    columns = ['time', 'linear_v', 'angular_omega_rads']
                elif key == "control_delta":
                    columns = ['time', 'delta_rad']
                elif key == "vel_cmd":    # For MPC linear velocity commands
                    columns = ['time', 'v_left_cmd', 'v_right_cmd']
                else:    # Fallback or skip unknown keys
                    # print(f"Warning: Unknown log key '{key}' or unhandled data structure.")
                    processed[key] = pd.DataFrame(data_entries)    # Try generic conversion
                    continue

                # Ensure all entries have the correct number of columns before creating DataFrame
                # This handles cases where some entries might be malformed (e.g. MPC vel_cmd if not active)
                valid_entries = [entry for entry in data_entries if len(entry) == len(columns)]
                if valid_entries:
                    processed[key] = pd.DataFrame(valid_entries, columns=columns)
                else:
                    processed[key] = pd.DataFrame(columns=columns)    # Empty DF with correct columns

            self.processed_logs.append(processed)

    def plot_trajectories(self, reference_path_x=None, reference_path_y=None, title_suffix=""):
        """Plots the vehicle trajectories from all logs against an optional reference path."""
        plt.figure(figsize=(10, 8))
        if reference_path_x is not None and reference_path_y is not None:
            plt.plot(reference_path_x, reference_path_y, "k-", label="Reference Path", linewidth=2)

        for i, log in enumerate(self.processed_logs):
            df_pose = log.get("vehicle_pose")
            if df_pose is not None and not df_pose.empty:
                plt.plot(df_pose['x'], df_pose['y'], "--", label=f"Trajectory: {self.labels[i]}")

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Vehicle Trajectory Comparison {title_suffix}")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_rpms(self, title_suffix=""):
        """
        Plots reference vs actual RPMs for left and right wheels in subplots.
        """
        num_logs = len(self.processed_logs)
        # Create a figure with 2 subplots (for left and right wheels)
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"RPM Profiles Comparison {title_suffix}", fontsize=16)

        for i, log in enumerate(self.processed_logs):
            label_prefix = self.labels[i]

            # Plot Left Wheel RPMs on axs[0]
            df_rpm_ref = log.get("rpm_ref")
            if df_rpm_ref is not None and not df_rpm_ref.empty and 'left' in df_rpm_ref.columns:
                axs[0].plot(df_rpm_ref['time'], df_rpm_ref['left'], linestyle='--', label=f"{label_prefix} Left Ref")

            df_rpms_actual = log.get("rpms")
            if df_rpms_actual is not None and not df_rpms_actual.empty and 'left' in df_rpms_actual.columns:
                axs[0].plot(df_rpms_actual['time'], df_rpms_actual['left'], linestyle='-', label=f"{label_prefix} Left Actual")

            # Plot Right Wheel RPMs on axs[1]
            if df_rpm_ref is not None and not df_rpm_ref.empty and 'right' in df_rpm_ref.columns:
                axs[1].plot(df_rpm_ref['time'], df_rpm_ref['right'], linestyle='--', label=f"{label_prefix} Right Ref")

            if df_rpms_actual is not None and not df_rpms_actual.empty and 'right' in df_rpms_actual.columns:
                axs[1].plot(df_rpms_actual['time'], df_rpms_actual['right'], linestyle='-', label=f"{label_prefix} Right Actual")

        axs[0].set_ylabel("Left Wheel RPM")
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        axs[1].set_ylabel("Right Wheel RPM")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])    # Adjust layout for suptitle
        plt.show()

    def plot_motor_commands(self, title_suffix=""):
        """
        Plots the motor command signals (e.g., voltage or normalized effort)
        for left and right motors in subplots.
        """
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Motor Command Signals Comparison {title_suffix}", fontsize=16)

        for i, log in enumerate(self.processed_logs):
            label_prefix = self.labels[i]
            df_motor_cmd = log.get("motor_cmd")

            if df_motor_cmd is not None and not df_motor_cmd.empty:
                if 'left' in df_motor_cmd.columns:
                    axs[0].plot(df_motor_cmd['time'], df_motor_cmd['left'], linestyle='-', label=f"{label_prefix} Left Cmd")
                if 'right' in df_motor_cmd.columns:
                    axs[1].plot(df_motor_cmd['time'], df_motor_cmd['right'], linestyle='-', label=f"{label_prefix} Right Cmd")

        axs[0].set_ylabel("Left Motor Command")
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        axs[1].set_ylabel("Right Motor Command")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_steering_angles(self, title_suffix=""):
        """Plots the commanded steering angle (delta)."""
        plt.figure(figsize=(12, 6))
        for i, log in enumerate(self.processed_logs):
            label_prefix = self.labels[i]
            df_delta = log.get("control_delta")
            if df_delta is not None and not df_delta.empty and 'delta_rad' in df_delta.columns:
                plt.plot(df_delta['time'], np.rad2deg(df_delta['delta_rad']), label=f"{label_prefix} Steering Angle [deg]")

        plt.xlabel("Time (s)")
        plt.ylabel("Steering Angle (degrees)")
        plt.title(f"Steering Angle (Delta) Profile {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_vehicle_speeds(self, target_linear_speed=None, title_suffix=""):
        """Plots linear and angular vehicle speeds."""
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Vehicle Speeds Profile Comparison {title_suffix}", fontsize=16)

        for i, log in enumerate(self.processed_logs):
            label_prefix = self.labels[i]
            df_speeds = log.get("vehicle_speeds")
            if df_speeds is not None and not df_speeds.empty:
                if 'linear_v' in df_speeds.columns:
                    axs[0].plot(df_speeds['time'], df_speeds['linear_v'], label=f"{label_prefix} Linear Speed (m/s)")
                if 'angular_omega_rads' in df_speeds.columns:
                    axs[1].plot(df_speeds['time'], np.rad2deg(df_speeds['angular_omega_rads']), label=f"{label_prefix} Angular Velocity (deg/s)")

        if target_linear_speed is not None:
            axs[0].axhline(target_linear_speed, color='r', linestyle='--', label=f'Target Speed ({target_linear_speed} m/s)')

        axs[0].set_ylabel("Linear Speed (m/s)")
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        axs[1].set_ylabel("Angular Velocity (deg/s)")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    # You can add more plotting methods as needed (e.g., for specific MPC costs, errors, etc.)
