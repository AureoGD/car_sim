import numpy as np
from base_controller import BaseController

class PurePursuitController(BaseController):
    def __init__(self, path_x, path_y, lookahead_distance=1.0):
        self.path_x = np.array(path_x)
        self.path_y = np.array(path_y)
        self.Ld = lookahead_distance

    def find_lookahead_point(self, x, y, theta):
        for i in range(len(self.path_x)):
            dx = self.path_x[i] - x
            dy = self.path_y[i] - y
            dist = np.hypot(dx, dy)

            # Check both distance and forward direction
            heading_vector = np.array([np.cos(theta), np.sin(theta)])
            path_vector = np.array([dx, dy])
            if dist >= self.Ld and np.dot(heading_vector, path_vector) > 0:
                return self.path_x[i], self.path_y[i]

        return self.path_x[-1], self.path_y[-1]

    def compute_steering_angle(self, x, y, theta, v, L):
        tx, ty = self.find_lookahead_point(x, y, theta)
        dx = tx - x
        dy = ty - y
        alpha = np.arctan2(dy, dx) - theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # wrap to [-pi, pi]
        curvature = 2 * np.sin(alpha) / self.Ld
        delta = np.arctan(L * curvature)
        return delta, (tx, ty)
