import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Often useful with the plotter, though plotter handles it internally

# Assuming these files exist and controllers are updated for BaseController
from path_generator import PathGenerator
from pure_pursuit_controller import PurePursuitController
from stanley_controller import StanleyController
from mpc_controller import MPCController  # Needs refactor for new BaseController
from ackermann_model import AckermannSlipModel
from simulator import Simulator  # Assuming your latest simulator is in simulator.py
from plot_data import SimulationPlotter  # Import the new plotter class

# from visualize_sim import VisualizerSim # Kept commented


def run_simulation(use_mechanical_differential_model, path_x, path_y,
                   path_theta, control_method, v_car_ref_sim, T_sim):
    """
    Runs a single simulation with the specified parameters.
    """
    # 1. Create the vehicle model
    # AckermannSlipModel uses 'use_mechanical_differential'
    model = AckermannSlipModel(
        use_mechanical_differential=use_mechanical_differential_model,
        slip_gain=1)

    # 2. Instantiate the selected controller
    # (Ensuring controllers are updated to use the new BaseController structure)
    if control_method == "pure_pursuit":
        controller = PurePursuitController(model=model,
                                           path_x=path_x,
                                           path_y=path_y,
                                           lookahead_distance=0.5)
    elif control_method == "stanley":
        controller = StanleyController(model=model,
                                       path_x=path_x,
                                       path_y=path_y,
                                       path_theta=path_theta,
                                       k=0.001)
    elif control_method == "mpc":
        # MPCController is designed to work with BaseController if instantiated correctly.
        # Provide model, path, and MPC-specific parameters.
        path_v_ref = np.ones(
            len(path_x)) * v_car_ref_sim  # Example target speed for the path

        # Example Q and R diagonal values for MPC cost function
        # q_diag: [x_err, y_err, theta_err, v_err]
        # r_diag: [delta_v_left, delta_v_right, delta_steer_angle]
        q_diag_mpc = [5.0, 5.0, 1.5, 1.0, 5.0]
        r_diag_mpc = [1.0, 1.0, 1.5]

        controller = MPCController(
            model=model,
            path_x=path_x,
            path_y=path_y,
            path_theta=path_theta,
            ref_v=0.75,  # Pass reference velocities for the path
            dt=
            0.1,  # MPC sample time, consider aligning with sim.dt_path_controller
            horizon=10,  # Prediction horizon (N)
            control_horizon_m=5,  # Control horizon (M)
            use_differential=
            use_mechanical_differential_model,  # For slip constraint
            q_diag=q_diag_mpc,  # State error costs
            r_diag=r_diag_mpc  # Control input change costs
        )
    else:
        raise ValueError(f"Unknown control method: {control_method}")

    # 3. Instantiate the Simulator
    sim = Simulator(
        model=model,
        controller=controller,
        use_velocity_controller=False,  # Constant target speed for non-MPC
        x0=0.0,
        y0=0.0,
        theta0=0.0,
        v_car_ref=v_car_ref_sim,
        T=T_sim,
        dt=0.001,  # Using Simulator's default base dt
        # Other dt values (dt_path_controller, etc.) will use Simulator's defaults
    )

    # 4. Run the simulation
    sim.run()

    # 5. Return the simulation log
    return sim.system_log


if __name__ == "__main__":
    CONTROL_METHOD = "mpc"  # Options: "pure_pursuit", "stanley" (MPC needs refactor)
    V_CAR_REF_MAIN = 0.75
    T_MAIN = 32.5
    # Parameter for AckermannSlipModel's differential behavior
    USE_MECHANICAL_DIFFERENTIAL_IN_MODEL = False  # Test without the ideal differential effect

    # Create reference path
    path_gen = PathGenerator(start_pos=(0, 0), start_theta=0)
    path_gen.add_straight(length=5)
    path_gen.add_curve(radius=3, angle_deg=60)
    path_gen.add_straight(length=10)
    path_gen.add_curve(radius=4, angle_deg=-60)
    path_gen.add_straight(length=5)
    path_x, path_y, path_theta = path_gen.get_path()

    print(
        f"Running simulation with: {CONTROL_METHOD.upper()}, "
        f"Mechanical Differential in Model: {USE_MECHANICAL_DIFFERENTIAL_IN_MODEL}, "
        f"Target Speed: {V_CAR_REF_MAIN} m/s, Duration: {T_MAIN}s")

    # Run a single simulation for now
    log_data_run1 = run_simulation(
        use_mechanical_differential_model=USE_MECHANICAL_DIFFERENTIAL_IN_MODEL,
        path_x=path_x,
        path_y=path_y,
        path_theta=path_theta,
        control_method=CONTROL_METHOD,
        v_car_ref_sim=V_CAR_REF_MAIN,
        T_sim=T_MAIN)
    print("Simulation finished. Initializing plotter...")

    # # Use the SimulationPlotter
    # # For a single run:
    # plotter = SimulationPlotter(
    #     log_data_list=[log_data_run1],
    #     labels=[
    #         f"{CONTROL_METHOD} (Diff_Model: {USE_MECHANICAL_DIFFERENTIAL_IN_MODEL})"
    #     ])

    # --- Example of running two simulations for comparison ---
    print("\nRunning second simulation with different parameters...")
    log_data_run2 = run_simulation(
        use_mechanical_differential_model=
        True,  # Compare with ideal differential
        path_x=path_x,
        path_y=path_y,
        path_theta=path_theta,
        control_method=CONTROL_METHOD,
        v_car_ref_sim=V_CAR_REF_MAIN,
        T_sim=T_MAIN)
    print("Second simulation finished. Initializing plotter for comparison...")
    plotter_comparison = SimulationPlotter(
        log_data_list=[log_data_run1, log_data_run2],
        labels=[
            f"{CONTROL_METHOD} (Diff_Model: {USE_MECHANICAL_DIFFERENTIAL_IN_MODEL})",
            f"{CONTROL_METHOD} (Diff_Model: True)"
        ])
    # # Now use plotter_comparison for plots
    plotter = plotter_comparison  # Use the comparison plotter for subsequent calls
    # --- End of comparison example ---

    print("Generating plots...")
    plotter.plot_trajectories(reference_path_x=path_x, reference_path_y=path_y)
    plotter.plot_rpms()
    plotter.plot_motor_commands(
    )  # Plots the output of the MotorPID controllers
    plotter.plot_steering_angles()
    plotter.plot_vehicle_speeds(target_linear_speed=V_CAR_REF_MAIN)
    plotter.plot_heading_over_time()
    plotter.plot_mpc_du()
    print("All plots generated.")

    # Visualize in PyBullet (kept commented as in original)
    # viz = VisualizerSim(path_x, path_y)
    # if "vehicle_pose" in log_data_run1 and log_data_run1["vehicle_pose"]:
    #     # VisualizerSim might need a list of (x,y,theta) tuples or similar
    #     # Assuming log_data_run1["vehicle_pose"] is [[t,x,y,th,b], ...]
    #     traj_to_viz = [(entry[1], entry[2], entry[3]) for entry in log_data_run1["vehicle_pose"]]
    #     viz.add_car_trajectory(traj_to_viz, color=[0, 0, 1, 1])
    # viz.render(sleep_time=0.01, record_video=False, follow_car_index=0)
