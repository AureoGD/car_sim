from path_generator import PathGenerator
from pure_pursuit_controller import PurePursuitController
from stanley_controller import StanleyController
from ackermann_model import AckermannKinematics
from simulator import Simulator
from result_plotter import ResultPlotter
from velocity_controller import VelocityPID


def main():
    # ─────────────────────────────
    # Configurable Parameters
    # ─────────────────────────────
    controller_type = "stanley"  # pure_pursuit or "stanley"
    use_velocity_pid = True
    lookahead_distance = 1.0
    stanley_gain = 0.1
    v_ref = 1.0
    dt = 0.05
    T = 20.0

    # ─────────────────────────────
    # Path creation
    # ─────────────────────────────
    path = PathGenerator(start_pos=(0, 0), start_theta=0)
    path.add_straight(5)
    path.add_curve(radius=5, angle_deg=90)
    path.add_straight(5)
    path.add_curve(radius=3, angle_deg=-90)
    path.add_straight(5)

    path_x, path_y, path_theta = path.get_path()

    # ─────────────────────────────
    # Controller Selection
    # ─────────────────────────────
    if controller_type == "pure_pursuit":
        controller = PurePursuitController(path_x, path_y, lookahead_distance)
    elif controller_type == "stanley":
        controller = StanleyController(path_x, path_y, path_theta, k=stanley_gain)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    velocity_ctrl = VelocityPID(kp=0.025, ki=0.01, kd=0.00001)

    # ─────────────────────────────
    # Run Simulation
    # ─────────────────────────────
    model = AckermannKinematics()
    sim = Simulator(model, controller, x0=0.0, y0=0.0, theta0=0.0,
                    v_ref=v_ref, dt=dt, T=T, use_velocity_pid=False,
                    velocity_controller=velocity_ctrl)
    sim.run()
    traj_x, traj_y, traj_theta, deltas, rpms, velocities = sim.get_trajectory()

    # ─────────────────────────────
    # Plot Results
    # ─────────────────────────────
    plotter = ResultPlotter(path_x, path_y)
    plotter.plot_trajectory(traj_x, traj_y, traj_theta)
    plotter.plot_steering(deltas, dt=dt)
    plotter.plot_velocity_rpm(velocities, rpms, dt=dt)


if __name__ == "__main__":
    main()
