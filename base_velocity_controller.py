class BaseVelocityController:
    def control(self, v_ref, v_meas, dt):
        raise NotImplementedError("This method should be implemented by a velocity controller subclass.")

    def reset(self):
        raise NotImplementedError("This method should be implemented by a velocity controller subclass.")