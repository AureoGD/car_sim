import numpy as np
from base_controller import BaseController


class PurePursuitController(BaseController):

    def __init__(self, model, path_x, path_y, lookahead_distance=1.0, **kwargs):
        """
        Initializes the Pure Pursuit controller.

        Args:
            model: The vehicle model object (to access parameters like wheelbase L).
            path_x (list or np.array): Reference path x-coordinates.
            path_y (list or np.array): Reference path y-coordinates.
            lookahead_distance (float): The lookahead distance (Ld) for the controller.
            **kwargs: Additional keyword arguments for the BaseController or other extensions.
        """
        # Call the BaseController's __init__ method
        # Pass the model, common path elements, and any unconsumed kwargs
        super().__init__(model, path_x=path_x, path_y=path_y, **kwargs)

        self.control_name = "pure_pursuit"
        self.Ld = lookahead_distance    # Pure Pursuit specific parameter

        # Vehicle wheelbase (self.L) is inherited from BaseController (taken from model)
        # Path arrays (self._path_x, self._path_y) are also handled by BaseController

    def _find_lookahead_point(self) -> tuple:
        """
        Finds the lookahead point on the path based on the current vehicle state.
        Uses self.current_x, self.current_y, self.current_theta, self._path_x, self._path_y, self.Ld.

        Returns:
            tuple: (tx, ty) coordinates of the lookahead target point.
        """
        if self._path_x.size == 0 or self._path_y.size == 0:
            # print("Warning: Path is empty in PurePursuitController. Returning current position as target.")
            return self.current_x, self.current_y    # Fallback if no path

        # Search for the lookahead point
        # A more efficient search could start from the previously found closest point,
        # but a full scan is robust for now.
        for i in range(len(self._path_x) - 1, -1, -1):    # Search backwards to find last valid point
            dx_path_point = self._path_x[i] - self.current_x
            dy_path_point = self._path_y[i] - self.current_y
            dist_to_point = np.hypot(dx_path_point, dy_path_point)

            if dist_to_point <= self.Ld:
                # Check if the point is in front of the vehicle
                # Vector to path point relative to car's heading
                point_angle_in_vehicle_frame = np.arctan2(dy_path_point, dx_path_point) - self.current_theta

                # If the point is roughly within +/- 90 degrees in front
                if np.abs(np.arctan2(np.sin(point_angle_in_vehicle_frame), np.cos(point_angle_in_vehicle_frame))) <= (np.pi / 2):
                    # Now search forward from this point to find the intersection or closest point to Ld
                    for j in range(i, len(self._path_x)):
                        dx_segment = self._path_x[j] - self.current_x
                        dy_segment = self._path_y[j] - self.current_y
                        dist_segment = np.hypot(dx_segment, dy_segment)
                        if dist_segment >= self.Ld:
                            # Simple approach: return this point.
                            # More advanced: interpolate along the segment (path[j-1] to path[j])
                            # to find the exact point at distance Ld.
                            # For now, taking the first point beyond or at Ld found this way.
                            return self._path_x[j], self._path_y[j]
                    # If loop finishes, means all remaining points are beyond Ld but were "in front" initially.
                    # Or if we reached end of path and it was still within Ld.
                    return self._path_x[-1], self._path_y[-1]    # Return last point

        # Fallback: If no point is found within Ld while searching backwards (e.g., car far from path start)
        # or if all points are behind, return the last point of the path as a simple fallback.
        # A more robust fallback might be the closest point in front if no intersection is found.
        # For now, this relies on the loop above finding a suitable candidate.
        # If the above loop doesn't return, it means all points were > Ld or behind.
        # Let's try a simpler forward scan again for general cases or if the backward scan fails.
        for i in range(len(self._path_x)):
            dx = self._path_x[i] - self.current_x
            dy = self._path_y[i] - self.current_y
            dist = np.hypot(dx, dy)

            heading_vector = np.array([np.cos(self.current_theta), np.sin(self.current_theta)])
            path_vector = np.array([dx, dy])
            if dist >= self.Ld and np.dot(heading_vector, path_vector) > 0:
                return self.path_x[i], self.path_y[i]

        # If still no point found (e.g. path is very short, or car is past the path)
        # return the last point of the path
        if self._path_x.size > 0:
            return self._path_x[-1], self._path_y[-1]
        else:    # Path is truly empty
            return self.current_x, self.current_y

    def compute_control(self) -> dict:
        """
        Computes the steering angle using the Pure Pursuit algorithm.
        Uses vehicle state (self.current_x, self.current_y, self.current_theta),
        wheelbase (self.L), and path (self._path_x, self._path_y).
        The current vehicle speed (self.current_v) is available but not used
        by the classic Pure Pursuit steering law.

        Returns:
            dict: A dictionary containing the computed steering angle ('delta')
                  and the lookahead target point ('debug_target_point').
        """
        if self._path_x is None or self._path_x.size == 0:
            # print("Warning: Path not set for Pure Pursuit controller. Outputting zero steering.")
            return {'delta': 0.0, 'debug_target_point': (self.current_x, self.current_y)}

        tx, ty = self._find_lookahead_point()

        # Calculate alpha: the angle to the lookahead point in the vehicle's coordinate frame
        dx_to_target = tx - self.current_x
        dy_to_target = ty - self.current_y

        alpha = np.arctan2(dy_to_target, dx_to_target) - self.current_theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))    # Wrap alpha to [-pi, pi]

        # Pure Pursuit steering law
        # Curvature = 2 * sin(alpha) / Ld
        # delta = arctan(L * curvature)
        # Add a small epsilon to Ld to prevent division by zero if Ld could be zero,
        # though Ld is typically a positive constant.
        if abs(self.Ld) < 1e-6:    # Avoid division by zero if Ld is accidentally zero
            curvature = 0.0
        else:
            curvature = 2 * np.sin(alpha) / self.Ld

        delta = np.arctan(self.L * curvature)    # self.L is the wheelbase from BaseController

        return {'delta': delta, 'debug_target_point': (tx, ty)}
