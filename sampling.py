import numpy as np
from scene_objects import Point3d


class FeatureTracker:

    def __init__(self) -> None:
        pass

    def track(self, points):
        pass

class Projection:

    def __init__(self, u, v, cam_id, p_id) -> None:
        """
        Constructor
        Args:
            u (float): u-coordinate of pixel
            v (float): v-coordinate of pixel
            cam_id (int): index of camera on which the projection was made
            p_id (int): index of projected 3D point
        """
        self.u = u
        self.v = v
        self.cam_id = cam_id
        self.p_id = p_id

class DataSampler:

    def __init__(self, pps=10) -> None:
        """
        Samples random data from the scene.
        Args:
            pps (int, optional): Number of points per plane to sample from. Defaults to 10.
        """
        self.pps = pps

    @staticmethod
    def sample_from_plane(plane):
        """
        Samples random point from given plane
        Args:
            plane (Plane): plane from which points are sampled.

        Returns:
            list: x, y, z coordinates of the point.
        """
        # get random coefficients [0, 1]
        a = np.random.random()
        b = np.random.random()
        # sample point as a weighted sum of two vectors constructing a rectangle
        v1, v2 = plane.points[:, 1:2] - plane.points[:, 0:1], plane.points[:, 3:4] - plane.points[:, 0:1]
        p = a * v1 + b * v2 + plane.points[:, 0:1]
        return [p[0, 0], p[1, 0], p[2, 0]]

    def sample(self, scene, cameras):
        """
        Samples random points from the scene and projects them on cameras.
        Args:
            scene (Scene): instance containing of all 3D objects in the scene
            cameras (Camera): list of all cameras looking at the scene

        Returns:
            tuple: projections of sampled 3D points and sampled 3D points
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
        projections_list = []
        points_list = [Point3d(points[index][0, 0], points[index][1, 0], points[index][2, 0]) 
                        for index in range(points.shape[0])]
        
        for cam_id, cam in enumerate(cameras):
            for p_id, point in enumerate(points):
                projected_point = cam.project_point(point)
                u = projected_point[0, 0]
                v = projected_point[1, 0]
                if u is not None and v is not None:
                    projections_list.append(Projection(u, v, cam_id, p_id))
        return projections_list, points_list
