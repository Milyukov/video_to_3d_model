import cv2
import numpy as np
import tqdm

class Camera:

    def __init__(self, fx, fy, cx, cy, width, height, x, y, z, roll, pitch, yaw) -> None:
        """Initialize pinhole camera

        Args:
            fx (double): _description_
            fy (double): _description_
            cx (double): _description_
            cy (double): _description_
            width (int): image width, pixels
            height (int): image height, pixels
            x (double): x-coordiante of camera optical center in world axes, m
            y (double): y-coordiante of camera optical center in world axes, m
            z (double): z-coordiante of camera optical center in world axes, m
            roll (double): roll angle of camera rotation in world axes, degress
            pitch (double): pitch angle of camera rotation in world axes, degress
            yaw (double): yaw angle of camera rotation in world axes, degress
        """
        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
        self.K_inv = np.linalg.inv(self.K)
        self.width = width
        self.height = height
        self.center = np.array([[
            x, y, z
        ]]).T
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        roll_rad = np.deg2rad(roll)
        R_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0]
        ])
        R_picth = np.array([
          [np.cos(pitch_rad), 0.0, -np.sin(pitch_rad)],
          [0.0, 1.0, 0.0],
          [np.sin(pitch_rad), 0.0, np.cos(pitch_rad)]  
        ])
        R_roll = np.array([
          [1.0, 0.0, 0.0],
          [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
          [0.0, np.sin(roll_rad), np.cos(roll_rad)]  
        ])

        self.R = np.dot(R_yaw, np.dot(R_picth, R_roll))

        self.max_distance = 20

    def project_points(self):
        pass
    
    def eval_intensity(self, distance):
        if distance > self.max_distance:
            return 0
        return int(255 * distance / self.max_distance)

    def eval_distance(self, point):
        dist = self.center - point
        dist = np.linalg.norm(dist)
        return dist

    def render_scene(self, scene):
        """Render scene using ray casting approach

        Args:
            scene (Scene): instance of class Scene, describing scene in 3D as a list of objects
        """
        rendered_scene = np.zeros((self.height, self.width), np.uint8)
        # for each pixel
        for y in tqdm.tqdm(range(self.height)):
            for x in range(self.width):
                # reconstruct a ray
                pix_hom = np.array([[x, y, 1.0]]).T
                ray = np.dot(self.K_inv, pix_hom)
                ray = np.dot(self.R, ray)
                # intersect a ray with a scene
                intersection_points = []
                for obj in scene.objects:
                    intersection_points.extend(obj.find_intersections(ray, self.center))
                if len(intersection_points) > 0:
                    # find closest point of intersection
                    min_dist = np.inf
                    for point in intersection_points:
                        dist = self.eval_distance(point)
                        if dist < min_dist:
                            min_dist = dist
                    # set pixel value to corresponding value
                    if min_dist != np.inf:
                        rendered_scene[y, x] = self.eval_intensity(min_dist)
        return rendered_scene[::-1, ::-1]

class Model3D:

    def __init__(self) -> None:
        pass

class Plane:

    def __init__(self, points, normal) -> None:
        self.points = points
        self.normal = normal if normal.shape == (3, 1) else np.expand_dims(normal, 1)
        self.d = np.abs(np.dot(np.mean(self.points, axis=1, keepdims=True).T, self.normal))

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
            [x - w / 2, y + h / 2, z + d / 2], # P0
            [x + w / 2, y + h / 2, z + d / 2], # P1
            [x + w / 2, y + h / 2, z - d / 2], # P2
            [x - w / 2, y + h / 2, z - d / 2], # P3
            [x - w / 2, y - h / 2, z + d / 2], # P4
            [x + w / 2, y - h / 2, z + d / 2], # P5
            [x + w / 2, y - h / 2, z - d / 2], # P6
            [x - w / 2, y - h / 2, z - d / 2]  # P7
        ]
        points = np.array(points).T
        # rotate points
        self.R = np.array([
            # first row
            [np.cos(yaw) * np.cos(pitch), 
            np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), 
            np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
            # second row
            [np.sin(yaw) * np.cos(pitch),
            np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
            np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
            # third row
            [-np.sin(pitch), 
            np.cos(pitch) * np.sin(roll), 
            np.cos(pitch) * np.cos(roll)]
        ])
        self.points = np.dot(self.R, points)
        # plane is defined by a point on it and a normal vector
        # so each plane is defined by normal vector and 4 points
        # estimate normal vector for each plane as a vector product of two edges constructing a plane
        self.top_plane = Plane(self.points[:, :4], np.cross(self.points[:, 0] - self.points[:, 1], self.points[:, 0] - self.points[:, 3]))
        self.bottom_plane = Plane(self.points[:, 4:], np.cross(self.points[:, 4] - self.points[:, 5], self.points[:, 4] - self.points[:, 7]))
        self.left_plane = Plane(self.points[:, [0, 3, 4, 7]], np.cross(self.points[:, 0] - self.points[:, 4], self.points[:, 0] - self.points[:, 3]))
        self.right_plane = Plane(self.points[:, [1, 2, 5, 6]], np.cross(self.points[:, 1] - self.points[:, 2], self.points[:, 1] - self.points[:, 5]))
        self.front_plane = Plane(self.points[:, [2, 3, 6, 7]], np.cross(self.points[:, 2] - self.points[:, 3], self.points[:, 2] - self.points[:, 6]))
        self.rare_plane = Plane(self.points[:, [0, 1, 4, 5]], np.cross(self.points[:, 0] - self.points[:, 1], self.points[:, 0] - self.points[:, 4]))

    @staticmethod
    def find_intersection(ray, p0, plane):
        # plane is parametrized as: ax + by + cz + d = 0
        # ray is parametrized as starting point P0 and direction V
        # shift along the ray to point of intersection is t (P = P0 + t * V)
        # t = -(P0 * N + d) / (V * N)
        # P = P0 + t * V
        denom = np.dot(ray.T, plane.normal)
        if denom != 0:
            t = -(np.dot(p0.T, plane.normal) + plane.d) / denom
            point = p0 + t * ray
            if point[0] > np.min(plane.points[0, :]) and point[0] < np.max(plane.points[0, :]):
                if point[1] > np.min(plane.points[1, :]) and point[1] < np.max(plane.points[1, :]):
                    return point
        return None

    def find_intersections(self, ray, p0):
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

class Scene:

    def __init__(self) -> None:
        """Initialize scene as a number of geometric promitives in 3D
        """
        self.objects = []
        #self.objects.append(Box(0, 0.5, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
        self.objects.append(Box(1, 0.25, -3.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0))

class FeatureTracker:

    def __init__(self) -> None:
        pass

    def track(self, points):
        pass

if __name__ == '__main__':
    # initialize scene
    scene = Scene()
    # initialize cameras and visualize rendered scene on each camera
    cameras = []
    frames = []
    for cam_num in range(1):
        cameras.append(Camera(500, 500, 480, 270, 960, 540, 0, 0.5, 7, 0, 0, 0))
        frames.append(cameras[cam_num].render_scene(scene))
        cv2.imshow("", frames[-1])
        cv2.waitKey(0)
    # sample points from the scene
    # project points from scene to each camera
    points = []
    for cam in cameras:
        points.append(cam.project_points())
    # run feature tracker and compare with true feature correspondences
    tracker = FeatureTracker()
    tracker.track(points)
    # run orientation estimation and compare with true orientations
    # run 3D reconstruction and compare with scene
    # estimate point cloud with semantic and instance infor from the ground truth scene
    # estimate 3D model and compare with ground truth model
    pass
