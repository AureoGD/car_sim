import numpy as np


class BaseController:
    """
    Abstract base class for vehicle controllers.

    This class defines a common interface for various controller implementations.
    Child controllers are expected to:
    1. Override `control_name`.
    2. Implement the `compute_control()` method to calculate control actions.
    3. Extract any specific initialization parameters from `**kwargs` in their `__init__`.
    """

    def __init__(self, model, path_x=None, path_y=None, path_theta=None, **controller_specific_kwargs):
        """
        Initializes the BaseController.

        Args:
            model: An object representing the vehicle model, from which common
                   physical parameters (e.g., wheelbase, wheel radius) can be accessed.
            path_x (list or np.array, optional): Initial reference path x-coordinates.
            path_y (list or np.array, optional): Initial reference path y-coordinates.
            path_theta (list or np.array, optional): Initial reference path orientations (radians).
                                                   Useful for controllers like Stanley or MPC.
            **controller_specific_kwargs: Catches additional keyword arguments,
                                          allowing child classes to define and use their
                                          own specific initialization parameters (e.g., gains).
        """
        # Store common vehicle constants from the model object
        self.L = model.L    # Example: wheelbase
        self.b = model.b    # Example: half track width (if available and needed)
        self.r = model.r    # Example: wheel radius (if available and needed)

        # Initialize vehicle state variables (will be updated by `update_states`)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_v = 0.0    # Current vehicle speed

        # Initialize path variables. Can be set initially or via the `set_path` method.
        # Using '_' prefix to indicate they are primarily managed by set_path or init.
        self._path_x = np.array(path_x) if path_x is not None else np.array([])
        self._path_y = np.array(path_y) if path_y is not None else np.array([])
        self._path_theta = np.array(path_theta) if path_theta is not None else np.array([])
        self._path_v = np.array([])    # For reference path velocities, if used

        self.control_name = "BaseController"    # Child classes should override this

        # Child classes are responsible for processing `controller_specific_kwargs`
        # Example: In a child class __init__:
        #   self.my_gain = controller_specific_kwargs.get('my_gain_param', default_value)
        #   super().__init__(model, path_x, path_y, path_theta, **remaining_kwargs_for_other_bases)

    def set_path(self, path_x: np.ndarray, path_y: np.ndarray, path_theta: np.ndarray = None, path_v: np.ndarray = None):
        """
        Sets or updates the reference path for the controller.

        This is useful if the path is not provided at initialization or if it needs
        to be changed dynamically during operation.

        Args:
            path_x (np.array): Reference path x-coordinates.
            path_y (np.array): Reference path y-coordinates.
            path_theta (np.array, optional): Reference path orientations (radians).
            path_v (np.array, optional): Reference path velocities (m/s).
        """
        self._path_x = np.array(path_x)
        self._path_y = np.array(path_y)
        self._path_theta = np.array(path_theta) if path_theta is not None else np.array([])
        self._path_v = np.array(path_v) if path_v is not None else np.array([])
        # print(f"Path updated for {self.control_name}.")

    def update_states(self, current_x: float, current_y: float, current_theta: float, current_v: float):
        """
        Updates the controller's internal knowledge of the vehicle's current state.

        This method MUST be called by the simulator before each call to `compute_control()`
        to ensure the controller is operating on the latest vehicle telemetry.

        Args:
            current_x (float): Current x-coordinate of the vehicle.
            current_y (float): Current y-coordinate of the vehicle.
            current_theta (float): Current orientation (yaw angle) of the vehicle (radians).
            current_v (float): Current longitudinal speed of the vehicle (m/s).
        """
        self.current_x = current_x
        self.current_y = current_y
        self.current_theta = current_theta
        self.current_v = current_v

    def compute_control(self) -> dict:
        """
        Computes the control action using the internally stored state (from `update_states`)
        and path (from `__init__` or `set_path`).

        This method MUST be implemented by child classes.

        Returns:
            dict: A dictionary of control commands. The specific keys and values
                  will depend on the child controller. Examples:
                  For lateral controllers (Stanley, Pure Pursuit):
                    {'delta': steering_angle_rad, 'debug_target_point': (tx, ty)}
                  For integrated controllers (MPC):
                    {'delta': steering_angle_rad,
                     'v_left_rads': target_left_wheel_angular_velocity_rad_s,
                     'v_right_rads': target_right_wheel_angular_velocity_rad_s}
                  It might also include other values like a target overall speed if applicable.
        """
        raise NotImplementedError("The compute_control() method must be implemented by child classes.")
