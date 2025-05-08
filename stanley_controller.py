from base_controller import BaseController
import numpy as np

class StanleyController(BaseController):
    def __init__(self, path_x, path_y, path_theta, k=1.0):
        self.path_x = np.array(path_x)
        self.path_y = np.array(path_y)
        self.path_theta = np.array(path_theta)
        self.k = k  # control gain

    def find_nearest_point(self, x, y):
        dists = np.hypot(self.path_x - x, self.path_y - y)
        min_idx = np.argmin(dists)
        return min_idx, dists[min_idx]

    def compute_steering_angle(self, x, y, theta, v, L):
        idx, cte = self.find_nearest_point(x, y)
        path_theta = self.path_theta[idx]

        # Heading error
        heading_error = path_theta - theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Cross-track error signed
        dx = self.path_x[idx] - x
        dy = self.path_y[idx] - y
        cross_track_sign = np.sign(np.sin(path_theta - np.arctan2(dy, dx)))
        cross_track_error = cte * cross_track_sign

        # Stanley control law
        steer_correction = np.arctan2(self.k * cross_track_error, v + 1e-5)
        delta = heading_error + steer_correction
        delta = np.arctan2(np.sin(delta), np.cos(delta))  # wrap
        return delta, (self.path_x[idx], self.path_y[idx])
