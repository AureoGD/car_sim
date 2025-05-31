from base_controller import BaseController
import numpy as np


class StanleyController(BaseController):

    def __init__(self, model, path_x, path_y, path_theta, k: float = 1.0, **kwargs):
        """
        Initializes the Stanley controller.

        Args:
            model: The vehicle model object (to access parameters like wheelbase L).
            path_x (list or np.array): Reference path x-coordinates.
            path_y (list or np.array): Reference path y-coordinates.
            path_theta (list or np.array): Reference path orientations (radians).
            k (float): Control gain for the cross-track error correction.
            **kwargs: Additional keyword arguments for the BaseController or other extensions.
        """
        # Call the BaseController's __init__ method.
        # Pass the model, common path elements (including path_theta which BaseController can store),
        # and any unconsumed kwargs.
        super().__init__(model, path_x=path_x, path_y=path_y, path_theta=path_theta, **kwargs)

        self.control_name = "stanley"
        self.k = k    # Stanley-specific control gain

        # Vehicle wheelbase (self.L) is inherited from BaseController.
        # Path arrays (self._path_x, self._path_y, self._path_theta) are handled by BaseController.

    def _find_nearest_path_point(self) -> tuple:
        """
        Finds the nearest point on the path to the vehicle's current position
        (self.current_x, self.current_y) and the cross-track error magnitude.

        Returns:
            tuple: (min_idx, cte_magnitude)
                min_idx (int): Index of the nearest point on the path.
                cte_magnitude (float): Magnitude of the cross-track error (distance to path).
        """
        if self._path_x is None or self._path_x.size == 0:
            # print("Warning: Path not set for Stanley controller. Cannot find nearest point.")
            # Fallback: if no path, effectively zero error relative to current spot, though this is degenerate.
            return 0, 0.0

        dists = np.hypot(self._path_x - self.current_x, self._path_y - self.current_y)
        min_idx = np.argmin(dists)
        return min_idx, dists[min_idx]

    def compute_control(self) -> dict:
        """
        Computes the steering angle using the Stanley method.
        Uses vehicle state (self.current_x, self.current_y, self.current_theta, self.current_v),
        wheelbase (self.L), path data (self._path_x, self._path_y, self._path_theta),
        and gain (self.k).

        For Stanley method, (current_x, current_y) is typically assumed to be the
        position of the front axle.

        Returns:
            dict: A dictionary containing the computed steering angle ('delta'),
                  the nearest path point ('debug_target_point'),
                  the signed cross-track error ('debug_cte'), and
                  the heading error ('debug_heading_error').
        """
        if self._path_x is None or self._path_x.size == 0 or \
           self._path_y is None or self._path_y.size == 0 or \
           self._path_theta is None or self._path_theta.size == 0:
            # print("Warning: Path (x, y, or theta) not properly set for Stanley controller. Outputting zero steering.")
            return {'delta': 0.0, 'debug_target_point': (self.current_x, self.current_y), 'debug_cte': 0.0, 'debug_heading_error': 0.0}

        idx, cte_magnitude = self._find_nearest_path_point()

        # Ensure path_theta has a valid entry for the found index
        if idx >= len(self._path_theta):
            # This might happen if path_theta is shorter than path_x/path_y after some path update
            # print("Warning: Nearest path index out of bounds for path_theta. Using last available theta.")
            path_theta_at_target = self._path_theta[-1] if len(self._path_theta) > 0 else self.current_theta
        else:
            path_theta_at_target = self._path_theta[idx]

        # 1. Heading Error (psi_error)
        # Difference between the path's orientation and the vehicle's current orientation
        heading_error = path_theta_at_target - self.current_theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))    # Normalize to [-pi, pi]

        # 2. Cross-Track Error (e_fa) - signed
        # Vector from vehicle's front axle (self.current_x, self.current_y) to the nearest path point
        dx_car_to_path = self._path_x[idx] - self.current_x
        dy_car_to_path = self._path_y[idx] - self.current_y

        # Sign of the cross-track error.
        # Positive if the path point is to the left of the vehicle's heading, negative if to the right.
        # Your original method: np.sign(np.sin(path_angle_at_target - angle_of_vector_to_target))
        # angle_of_vector_to_target = np.arctan2(dy_car_to_path, dx_car_to_path)
        # cross_track_sign = np.sign(np.sin(path_theta_at_target - angle_of_vector_to_target))

        # A common robust way to get signed CTE: project the error vector onto the path normal.
        # Path normal vector (points to the left of path tangent): (-sin(path_theta_at_target), cos(path_theta_at_target))
        # Error vector (from path point to car's front axle): (-dx_car_to_path, -dy_car_to_path)
        signed_cte = (-dx_car_to_path) * (-np.sin(path_theta_at_target)) + \
                     (-dy_car_to_path) * (np.cos(path_theta_at_target))
        # This directly gives the signed cross-track error.
        # If this value is positive, the car is to the left of the path.
        # If negative, car is to the right. Stanley typically wants e_fa positive if car is to right of path.
        # So, if car is to the left of path (path on its right), error is positive.
        # If car is to the right of path (path on its left), error is negative.
        # The term arctan(k*e_fa/v) aims to steer towards the path.
        # Let's re-check convention for Stanley: error e is positive if front axle is to the right of path.
        # So, if signed_cte above is positive (car to left of path), error for Stanley should be negative.
        cross_track_error = -signed_cte

        # Stanley control law: delta = heading_error + arctan(k * e_fa / (v + epsilon))
        # Add a small epsilon to current_v to prevent division by zero at low speeds.
        steer_correction_for_cte = np.arctan2(self.k * cross_track_error, self.current_v + 1e-5)

        delta = heading_error + steer_correction_for_cte
        delta = np.arctan2(np.sin(delta), np.cos(delta))    # Normalize to [-pi, pi]

        return {
            'delta': delta,
            'debug_target_point': (self._path_x[idx], self._path_y[idx]),
            'debug_cte': cross_track_error,
            'debug_heading_error': heading_error
        }
