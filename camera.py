import numpy as np
import tqdm
from utils import get_rotation_matrix

class Intrinsics:

    def __init__(self, fx, fy, cx, cy, width, height) -> None:
        """Intrinsic parameters of the camera

        Args:
            fx (double): focal distance in meters multiplied by number of pixels per meter for X-axis, pixels
            fy (double): focal distance in meters multiplied by number of pixels per meter for Y-axis, pixels
            cx (double): x-coordinate of optical center projection on matrix, pixels
            cy (double): y-coordinate of optical center projection on matrix, pixels
            width (int): image width, pixels
            height (int): image height, pixels
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
        self.K_inv = np.linalg.inv(self.K)
        self.width = width
        self.height = height

class Camera:


    def __init__(self, intrinsics, position, orientation) -> None:
        """Initialize pinhole camera

        Args:
            intrinsics(Intrinsics): intrinsic parameters of the camera
            position (tuple): coordiantes of camera optical center in world axes, m
            orientation (tuple): camera rotation in world axes, degress
        """
        self.intrinsics = intrinsics
        x, y, z = position
        self.center = np.array([[
            x, y, z
        ]]).T
        
        roll, pitch, yaw = orientation
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.R = get_rotation_matrix(roll, pitch, yaw)
        # maximum distance of visibility in meters
        self.max_distance = 40

    def project_point(self, point3d):
        point3d_cam = point3d + self.center
        point2d_hom = np.dot(self.intrinsics.K, np.dot(self.R, point3d_cam))
        point2d = np.array([point2d_hom[0] / point2d_hom[2], point2d_hom[1] / point2d_hom[2]])
        if point2d[0] < 0 or point2d[0] >= self.intrinsics.width:
            return np.array([[None, None]]).T
        if point2d[1] < 0 or point2d[1] >= self.intrinsics.height:
            return np.array([[None, None]]).T
        return point2d

    def project_points(self, points):
        projected_points = np.zeros((2, len(points)))
        for col, p in enumerate(points):
            projected_points[:, col:col + 1] = self.project_point(p)
        return projected_points.T
    
    def eval_intensity(self, distance):
        if distance > self.max_distance:
            return 0
        return int(255 * (self.max_distance - distance) / self.max_distance)

    def eval_distance(self, point):
        dist = self.center - point
        dist = np.linalg.norm(dist)
        return dist

    def render_scene(self, scene):
        """Render scene using ray casting approach

        Args:
            scene (Scene): instance of class Scene, describing scene in 3D as a list of objects
        """
        rendered_scene = np.zeros((self.intrinsics.height, self.intrinsics.width, 3), np.uint8)
        # for each pixel
        for y in tqdm.tqdm(range(self.intrinsics.height)):
            for x in range(self.intrinsics.width):
                # reconstruct a ray
                pix_hom = np.array([[x, y, 1.0]]).T
                ray = np.dot(self.intrinsics.K_inv, pix_hom)
                ray = np.dot(self.R.T, ray)
                # intersect a ray with a scene
                intersection_points = []
                for obj in scene.objects:
                    intersection_points.extend(obj.find_all_intersections(ray, self.center))
                if len(intersection_points) > 0:
                    # find closest point of intersection
                    min_dist = np.inf
                    for point in intersection_points:
                        dist = self.eval_distance(point)
                        if dist < min_dist:
                            min_dist = dist
                    # set pixel value to corresponding value
                    if min_dist != np.inf:
                        rendered_scene[y, x, :] = self.eval_intensity(min_dist)
        return rendered_scene[::-1, ::-1, :]
    
