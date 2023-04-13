import numpy as np
from scene_objects import Point3d


class FeatureTracker:

    def __init__(self) -> None:
        pass

    def track(self, points):
        pass

class Projection:

    def __init__(self, x, y, cam_id, p_id) -> None:
        self.x = x
        self.y = y
        self.cam_id = cam_id
        self.p_id = p_id

class DataSampler:

    def __init__(self, pps=10) -> None:
        """_summary_

        Args:
            pps (int, optional): Number of points per plane to sample from. Defaults to 10.
        """
        self.pps = pps

    @staticmethod
    def sample_from_plane(plane):
        # get random coefficients [0, 1]
        a = np.random.random()
        b = np.random.random()
        # sample point as a weighted sum of two vectors constructing a rectangle
        v1, v2 = plane.points[:, 1:2] - plane.points[:, 0:1], plane.points[:, 3:4] - plane.points[:, 0:1]
        p = a * v1 + b * v2 + plane.points[:, 0:1]
        return [p[0, 0], p[1, 0], p[2, 0]]

    def sample(self, scene, cameras):
        """_summary_
        Args:
            scene (Scene): instance containing of all 3D objects in the scene
            cameras (Camera): list of all cameras looking at the scene

        Returns:
            _type_: _description_
        """
        points = []
        # for each object in a scene
        for obj in scene.objects:
            # randomly sample 3D points on each side (except bottom and top)
            # for each side sample n points
            for _ in range(self.pps):
                p = self.sample_from_plane(obj.front_plane)
                points.append(p)
            for _ in range(self.pps):
                p = self.sample_from_plane(obj.right_plane)
                points.append(p)
            for _ in range(self.pps):
                p = self.sample_from_plane(obj.rare_plane)
                points.append(p)
            for _ in range(self.pps):
                p = self.sample_from_plane(obj.left_plane)
                points.append(p)
        
        # add projections to cameras
        points = np.array(points)
        points = np.expand_dims(points, axis=-1)
        # fill the structures
        projections_dict = {}
        points_dict = {index: Point3d(points[index][0, 0], points[index][1, 0], points[index][2, 0]) 
                        for index in range(points.shape[0])}
        cameras_dict = {}
        projections_number = 0
        for cam_id, cam in enumerate(cameras):
            cameras_dict[cam_id] = cam
            for p_id, point in enumerate(points):
                projected_point = cam.project_point(point)
                x = projected_point[0, 0]
                y = projected_point[1, 0]
                if x is not None and y is not None:
                    projections_dict[projections_number] = Projection(x, y, cam_id, p_id)
                    projections_number += 1
        return projections_dict, points_dict, cameras_dict