class BaseController:
    def compute_steering_angle(self, x, y, theta, v, L):
        raise NotImplementedError("This method should be implemented by child classes.")
