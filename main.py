# from path_generator import PathGenerator
# from pure_pursuit_controller import PurePursuitController
# from ackermann_model import AckermannSlipModel
# from simulator import Simulator
# from result_plotter import ResultPlotter


# if __name__ == "__main__":
#     # Create path
#     path = PathGenerator(start_pos=(0, 0), start_theta=0)
#     path.add_straight(5)
#     path.add_curve(radius=5, angle_deg=90)
#     path.add_straight(5)
#     path.add_curve(radius=3, angle_deg=-90)
#     path.add_straight(5)
#     path_x, path_y, path_theta = path.get_path()

#     # Initialize components
#     controller = PurePursuitController(path_x, path_y, lookahead_distance=0.5)
#     model = AckermannSlipModel(use_differential=True)
#     sim = Simulator(
#         model,
#         controller,
#         x0=0.0,
#         y0=0.0,
#         theta0=0.0,
#         dt=0.05,
#         T=3.0,
#     )

#     # Run simulation
#     sim.run()
#     traj_x, traj_y, traj_theta, deltas, rmps, velocities = sim.get_trajectory()

#     # Plot results
#     plotter = ResultPlotter(path_x, path_y)
#     plotter.plot_trajectory(traj_x, traj_y, traj_theta)
#     plotter.plot_steering(deltas, dt=0.05)

from path_generator import PathGenerator
from pure_pursuit_controller import PurePursuitController
from ackermann_model import AckermannSlipModel
from simulator import Simulator
import matplotlib.pyplot as plt


def run_simulation(use_differential, path_x, path_y, path_theta):
    controller = PurePursuitController(path_x, path_y, lookahead_distance=0.5)
    model = AckermannSlipModel(use_differential=use_differential, slip_gain=0.75)
    sim = Simulator(
        model,
        controller,
        x0=0.0,
        y0=0.0,
        theta0=0.0,
        dt=0.05,
        T=1.0,
    )
    sim.run()
    return sim.get_trajectory()


if __name__ == "__main__":
    # Create reference path
    path = PathGenerator(start_pos=(0, 0), start_theta=0)
    path.add_straight(5)
    path.add_curve(radius=5, angle_deg=90)
    path.add_straight(5)
    path.add_curve(radius=3, angle_deg=-90)
    path.add_straight(5)
    path_x, path_y, path_theta = path.get_path()

    # Run both simulations
    traj_with = run_simulation(True, path_x, path_y, path_theta)
    traj_without = run_simulation(False, path_x, path_y, path_theta)

    # Plot all in one figure
    plt.figure(figsize=(10, 6))
    plt.plot(path_x, path_y, "k", label="Reference Path", linewidth=2)
    plt.plot(traj_with[0], traj_with[1], "b--", label="With Differential")
    plt.plot(traj_without[0], traj_without[1], "r--", label="Without Differential")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectory Comparison: With vs. Without Differential")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
