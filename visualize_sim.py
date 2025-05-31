import math
import time
import pybullet as p
import pybullet_data
import numpy as np
import imageio


class VisualizerSim:
    def __init__(self, path_x, path_y, tile_spacing=0.2):
        self.path_x = path_x
        self.path_y = path_y
        self.tile_spacing = tile_spacing
        self.cars = []  # list of (traj_x, traj_y, traj_theta, color)

    def add_car_trajectory(self, traj_x, traj_y, traj_theta, color=[1, 0, 0, 1]):
        self.cars.append((traj_x, traj_y, traj_theta, color))

    def _compute_headings(self):
        headings = []
        for i in range(1, len(self.path_x)):
            dx = self.path_x[i] - self.path_x[i - 1]
            dy = self.path_y[i] - self.path_y[i - 1]
            headings.append(math.atan2(dy, dx))
        headings.append(headings[-1])  # Repeat last heading for consistency
        return headings

    def _draw_road_tiles(self, tile_half_length=0.2, tile_half_width=0.05, tile_thickness=0.01):
        headings = self._compute_headings()

        for x, y, theta in zip(self.path_x, self.path_y, headings):
            orientation = p.getQuaternionFromEuler([0, 0, theta])
            position = [x, y, tile_thickness]

            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[tile_half_length, tile_half_width, tile_thickness]
            )
            visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[tile_half_length, tile_half_width, tile_thickness],
                rgbaColor=[0.2, 0.2, 0.2, 1]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=position,
                baseOrientation=orientation
            )

    def render(self, sleep_time=0.01, record_video=False, video_path="car_path2.mp4", follow_car_index=None):
        # PyBullet setup
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Disable debug overlays
        for flag in [
            p.COV_ENABLE_GUI,
            p.COV_ENABLE_SHADOWS,
            p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        ]:
            p.configureDebugVisualizer(flag, 0)

        # Compute bounds to center the ground
        x_min, x_max = min(self.path_x), max(self.path_x)
        y_min, y_max = min(self.path_y), max(self.path_y)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        ground_size = max(x_max - x_min, y_max - y_min) + 10
        ground_thickness = 0.01

        ground_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[ground_size / 2, ground_size / 2, ground_thickness / 2],
            rgbaColor=[211/255, 211/255, 211/255, 0.8],
            specularColor=[0, 0, 0],
        )
        ground_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[ground_size / 2, ground_size / 2, ground_thickness / 2]
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_collision,
            baseVisualShapeIndex=ground_visual,
            basePosition=[center_x, center_y, ground_thickness / 2]
        )

        # Set initial camera
        p.resetDebugVisualizerCamera(
            cameraDistance=10.0,
            cameraYaw=45,
            cameraPitch=-40,
            cameraTargetPosition=[center_x, center_y, 0.5]
        )

        # Draw road
        self._draw_road_tiles()

        # Create cars
        car_ids = []
        box_size = (0.4, 0.2, 0.1)
        for traj_x, traj_y, traj_theta, color in self.cars:
            shape_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[s / 2 for s in box_size],
                rgbaColor=color
            )
            car_id = p.createMultiBody(
                baseMass=1,
                baseVisualShapeIndex=shape_id,
                basePosition=[traj_x[0], traj_y[0], box_size[2] / 2]
            )
            car_ids.append((car_id, traj_x, traj_y, traj_theta))

        # Setup video frame collection
        frames = []
        width, height = 640, 480

        # Animate all cars
        max_len = max(len(traj[1]) for traj in car_ids)
        for i in range(max_len):
            for car_id, traj_x, traj_y, traj_theta in car_ids:
                if i < len(traj_x):
                    x = traj_x[i]
                    y = traj_y[i]
                    theta = traj_theta[i]
                    pos = [x, y, box_size[2] / 2]
                    orn = p.getQuaternionFromEuler([0, 0, theta])
                    p.resetBasePositionAndOrientation(car_id, pos, orn)

            # Dynamic camera follow
            if follow_car_index is not None and 0 <= follow_car_index < len(car_ids):
                _, traj_x, traj_y, _ = car_ids[follow_car_index]
                if i < len(traj_x):
                    p.resetDebugVisualizerCamera(
                        cameraDistance=4.0,
                        cameraYaw=-30,
                        cameraPitch=-30,
                        cameraTargetPosition=[traj_x[i], traj_y[i], 0.5]
                    )

            p.stepSimulation()
            if record_video:
                _, _, rgb, _, _ = p.getCameraImage(width, height)
                img_np = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
                frames.append(img_np)
            time.sleep(sleep_time)

        # Save video
        if record_video:
            print(f"Captured {len(frames)} frames.")
            if frames:
                try:
                    imageio.mimsave(video_path, frames, fps=int(1 / sleep_time))
                    print(f"✅ Video saved to: {video_path}")
                except Exception as e:
                    print(f"❌ Failed to save video: {e}")
            else:
                print("⚠️ No frames captured. Video not saved.")

        p.disconnect()
