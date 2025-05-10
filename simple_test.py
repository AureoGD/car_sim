import matplotlib.pyplot as plt
from path_generator import PathGenerator
from pure_pursuit_controller import PurePursuitController
from stanley_controller import StanleyController
from ackermann_model import AckermannSlipModel
from simulatorv2 import Simulator
from visualize_sim import VisualizerSim

def run_simulation(
    use_differential, path_x, path_y, path_theta, control_method="pure_pursuit"
):
    if control_method == "pure_pursuit":
        controller = PurePursuitController(path_x, path_y, lookahead_distance=0.5)
    elif control_method == "stanley":
        controller = StanleyController(path_x, path_y, path_theta, k=0.001)
    model = AckermannSlipModel(use_differential=use_differential, slip_gain=1)
    sim = Simulator(model, controller, x0=0.0, y0=0.0, theta0=0.0, dt=0.01, T=30.0)
    sim.run()
    # sim.visualize_with_pybullet()
    return sim.get_system_states()


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
    log_with = run_simulation(
        True,
        path_x,
        path_y,
        path_theta,
    )

    log_without = run_simulation(
        False, path_x, path_y, path_theta, control_method="pure_pursuit"
    )

    # Plot all in one figure
    plt.figure(figsize=(10, 6))
    plt.plot(path_x, path_y, "k", label="Reference Path", linewidth=2)
    plt.plot(log_with["traj_x"], log_with["traj_y"], "b--", label="With Differential")
    plt.plot(
        log_without["traj_x"],
        log_without["traj_y"],
        "r--",
        label="Without Differential",
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectory Comparison: With vs. Without Differential")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualize in PyBullet
    viz = VisualizerSim(path_x, path_y)
    viz.add_car_trajectory(log_with["traj_x"], log_with["traj_y"], log_with["traj_theta"], color=[0, 0, 1, 1])  # Blue
    viz.add_car_trajectory(log_without["traj_x"], log_without["traj_y"], log_without["traj_theta"], color=[1, 0, 0, 1])  # Red
    viz.render(sleep_time=0.01, record_video=True, follow_car_index=1)