import numpy as np
from utils import get_rotation_matrix

class Point3d:

    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

class Plane:

    def __init__(self, points, normal) -> None:
        self.points = points
        self.normal = normal if normal.shape == (3, 1) else np.expand_dims(normal, 1)
        self.d = np.abs(np.dot(np.mean(self.points, axis=1, keepdims=True).T, self.normal))
        self.p0 = np.mean(self.points, axis=1, keepdims=True)

class Box:

    def __init__(self, x, y, z, w, h, d, roll, pitch, yaw) -> None:
        """Constructor of a class describing a box

        Args:
            x (double): x-coordinate of the center of a box, m
            y (double): y-coordinate of the center of a box, m
            z (double): z-coordinate of the center of a box, m
            w (double): width of the box, m
            h (double): height of the box, m
            d (double): depth of the box, m
            roll (double): roll angle, degrees
            pitch (double): pitch  angle, degrees
            yaw (double): yaw  angle, degrees
        """
        self.parameters = (x, y, z, w, h, d, roll, pitch, yaw)
        '''
                            Y         Z
                            ^        / 
                            |       /
                        P0 -|---------P1
                       / |  |     /  /|
                      /  |  |    /  / |
                    P3 ----------- P2 |
                     |   |  |  /   |  |
                     |  P4 -|------|--P5
                     | /    0------| /-------------------> X
                     |/            |/
                     P7 -----------P6
        
        '''
        # to generate planes from this parameters:
        # define points of edges intersections
        points = [
            [-w / 2,h / 2, d / 2], # P0
            [w / 2, h / 2, d / 2], # P1
            [w / 2, h / 2, -d / 2], # P2
            [-w / 2, h / 2, -d / 2], # P3
            [-w / 2, -h / 2, d / 2], # P4
            [w / 2, -h / 2, d / 2], # P5
            [w / 2, -h / 2, -d / 2], # P6
            [-w / 2, -h / 2, -d / 2]  # P7
        ]
        points = np.array(points).T
        # rotate points
        self.R = get_rotation_matrix(roll, pitch, yaw)
        self.points = np.dot(self.R, points) + np.array([[x, y, z]]).T
        # plane is defined by a point on it and a normal vector
        # so each plane is defined by normal vector and 4 points
        # estimate normal vector for each plane as a vector product of two edges constructing a plane
        self.top_plane = Plane(self.points[:, :4], np.cross(self.points[:, 0] - self.points[:, 1], self.points[:, 0] - self.points[:, 3]))
        self.bottom_plane = Plane(self.points[:, 4:], np.cross(self.points[:, 4] - self.points[:, 5], self.points[:, 4] - self.points[:, 7]))
        self.left_plane = Plane(self.points[:, [0, 4, 7, 3]], np.cross(self.points[:, 0] - self.points[:, 4], self.points[:, 0] - self.points[:, 3]))
        self.right_plane = Plane(self.points[:, [1, 2, 6, 5]], np.cross(self.points[:, 1] - self.points[:, 2], self.points[:, 1] - self.points[:, 5]))
        self.front_plane = Plane(self.points[:, [2, 3, 7, 6]], np.cross(self.points[:, 2] - self.points[:, 3], self.points[:, 2] - self.points[:, 6]))
        self.rare_plane = Plane(self.points[:, [0, 1, 5, 4 ]], np.cross(self.points[:, 0] - self.points[:, 1], self.points[:, 0] - self.points[:, 4]))

    @staticmethod
    def find_intersection(ray, l0, plane, debug=False):
        # Plane is parametrized as 3 or more points
        # ray is parametrized as starting point l0 and direction l
        # P is intersection point
        # shift along the ray to the point of intersection P is t (P = l0 + t * l)
        
        # Set plane points with l0 as origin
        p0 = plane.points[:, 0:1] - l0
        p1 = plane.points[:, 1:2] - l0
        p3 = plane.points[:, 3:4] - l0

        # Get sides of a rectangle as basis vectors
        r1 = p1 - p0
        r2 = p3 - p0

        # Form a linear system.
        # The solution is 
        # [
        #  r1 component - b1, 
        #  r2 component - b2, 
        #  t - shift along the ray to intersection point
        # ]
        A = np.concatenate((r1, r2, -ray), axis=1)

        try:
            x = np.linalg.solve(A, -p0)
        except np.linalg.LinAlgError:
            return None

        # Check if intersection point is within rectangle
        if 0 <= x[0] <= 1:
            if 0 <= x[1] <= 1:
                return l0 + ray*x[2]
        return l0 + ray*x[2,1] if debug else None

    def find_all_intersections(self, ray, p0):
        """Find all intersection points of box with given ray

        Args:
            ray (ndarray): (x, y, z)-components of a ray
        """
        # for each plane
        # check if there's intersection  with a ray inside 4 points range in point P
        intersections = []
        p = self.find_intersection(ray, p0, self.top_plane)
        if p is not None:
            intersections.append(p)
        p = self.find_intersection(ray, p0, self.bottom_plane)
        if p is not None:
            intersections.append(p)
        p = self.find_intersection(ray, p0, self.left_plane)
        if p is not None:
            intersections.append(p)
        p = self.find_intersection(ray, p0, self.right_plane)
        if p is not None:
            intersections.append(p)
        p = self.find_intersection(ray, p0, self.front_plane)
        if p is not None:
            intersections.append(p)
        p = self.find_intersection(ray, p0, self.rare_plane)
        if p is not None:
            intersections.append(p)
        return intersections