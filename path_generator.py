import numpy as np
import matplotlib.pyplot as plt


class PathSegment:
    def __init__(self, type, length, radius=None):
        self.type = type  # 'straight' or 'curve'
        self.length = length  # meters
        self.radius = radius  # only for curves (meters)


class PathGenerator:
    def __init__(self, start_pos=(0.0, 0.0), start_theta=0.0):
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.theta = start_theta
        self.xs = [self.x]
        self.ys = [self.y]
        self.thetas = [self.theta]

    def add_straight(self, length, resolution=0.1):
        steps = int(length / resolution)
        for _ in range(steps):
            self.x += resolution * np.cos(self.theta)
            self.y += resolution * np.sin(self.theta)
            self.xs.append(self.x)
            self.ys.append(self.y)
            self.thetas.append(self.theta)

    def add_curve(self, radius, angle_deg, resolution=0.1):
        angle_rad = np.deg2rad(angle_deg)
        arc_length = abs(radius * angle_rad)
        steps = int(arc_length / resolution)
        sign = np.sign(angle_rad)
        for _ in range(steps):
            self.theta += sign * resolution / radius
            self.x += resolution * np.cos(self.theta)
            self.y += resolution * np.sin(self.theta)
            self.xs.append(self.x)
            self.ys.append(self.y)
            self.thetas.append(self.theta)

    def get_path(self):
        return np.array(self.xs), np.array(self.ys), np.array(self.thetas)

    def save_to_csv(self, filename="path_data.csv"):
        data = np.column_stack((self.xs, self.ys, self.thetas))
        np.savetxt(filename, data, delimiter=",", header="x,y,theta", comments="")
        print(f"[INFO] Path saved to '{filename}' as CSV.")

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.xs, self.ys, label='Path')
        plt.quiver(self.xs[::20], self.ys[::20], np.cos(self.thetas[::20]), np.sin(self.thetas[::20]),
                   scale=30, width=0.0025, color='blue', label='Heading')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Generated Road Path')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
