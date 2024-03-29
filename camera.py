import numpy as np
import tqdm
from utils import get_rotation_matrix

class Intrinsics:

    def __init__(self, fu, fv, cu, cv, width, height) -> None:
        """
        Intrinsic parameters of the camera.\n
        u and v are horizontal and vertical pixel axes respectively.

        Args:
            fu (double): focal distance in meters multiplied by number of pixels per meter for u-axis, pixels
            fv (double): focal distance in meters multiplied by number of pixels per meter for v-axis, pixels
            cu (double): u-coordinate of optical center projection on matrix, pixels
            cv (double): v-coordinate of optical center projection on matrix, pixels
            width (int): image width, pixels
            height (int): image height, pixels
        """
        self.fu = fu
        self.fv = fv
        self.cu = cu
        self.cv = cv
        self.mat = np.array([
            [fu, 0.0, cu],
            [0.0, fv, cv],
            [0.0, 0.0, 1.0]
        ])
        self.mat_inv = np.linalg.inv(self.mat)
        self.width = width
        self.height = height

class Camera:


    def __init__(self, intrinsics, position, orientation, extrinsic_angles=False) -> None:
        """
        Initialize pinhole camera

        Args:
            intrinsics(Intrinsics): intrinsic parameters of the camera
            position (tuple): coordiantes of camera optical center in world axes in meters
            orientation (tuple): camera rotation in world axes in radians
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
        self.w2c_transform = get_rotation_matrix(roll, pitch, yaw, extrinsic=extrinsic_angles).T
        # maximum distance of visibility in meters
        self.max_distance = 40

    def project_point(self, point3d_w):
        """
        Projects 3D point in world coordinates on camera matrix.

        ^                   ^
        |                   |
        |                   |
        |                   C------>
        |                    \
        W--------->           \
         \
          \                             
           \                            P

        Let self.center be shift of C (camera center) relative to W (world origin).
        Let P be a point to project on camera. 
        Let it have point3d_w coordinates in world coordinate system 
        and point3d_c coordinates in camera coordinate system, then
        self.center + point3d_c = point3d_w -> point3d_c = self.center - point3d_w 
            
        Args:
            point3d_w (ndarray): 3D point in world coordinates

        Returns:
            ndarray: 2D coordinates of point3d_w projected on camera
        """
        point3d_c = point3d_w - self.center
        # point is shifted to camera coordinates system 
        # now we need to change the axes and then project it on camera
        point2d_hom = np.dot(self.intrinsics.mat, np.dot(self.w2c_transform, point3d_c))
        point2d = np.array([point2d_hom[0] / point2d_hom[2], point2d_hom[1] / point2d_hom[2]])
        if point2d[0] < 0 or point2d[0] >= self.intrinsics.width:
            return np.array([[None, None]]).T
        if point2d[1] < 0 or point2d[1] >= self.intrinsics.height:
            return np.array([[None, None]]).T
        return point2d

    def project_points(self, points_w):
        """
        Projects 3D points in world coordinates on camera matrix.
        Args:
            points_w (ndarray): array of 3D points in world coordinates.

        Returns:
            ndarray: array of projected points.
        """
        projected_points = np.zeros((2, len(points_w)))
        for col, p in enumerate(points_w):
            projected_points[:, col:col + 1] = self.project_point(p)
        return projected_points.T
    
    def eval_intensity(self, distance):
        """
        Evaluates intensity for rendered pixel based on the distance to projected 3D point (to emulate light).
        Args:
            distance (float): distance to projected 3D point.

        Returns:
            uint8: grayscale pixel value.
        """
        if distance > self.max_distance:
            return 0
        return int(255 * (self.max_distance - distance) / self.max_distance)

    def eval_distance(self, point_w):
        """
        Evaluates distance between point_w and camera center in world coordinate system.
        Args:
            point_w (ndarray): 3D point in world coordinate system.

        Returns:
            float: distance between point_w and camera center in world coordinate system.
        """
        dist = self.center - point_w
        dist = np.linalg.norm(dist)
        return dist

    def render_scene(self, scene):
        """
        Render scene using ray casting approach

        Args:
            scene (Scene): instance of class Scene, describing scene in 3D as a list of objects
        """
        rendered_scene = np.zeros((self.intrinsics.height, self.intrinsics.width, 3), np.uint8)
        # for each pixel
        for v in tqdm.tqdm(range(self.intrinsics.height)):
            for u in range(self.intrinsics.width):
                # reconstruct a ray in world axes
                pix_hom = np.array([[u, v, 1.0]]).T
                ray_c = np.dot(self.intrinsics.mat_inv, pix_hom)
                ray_w = np.dot(self.w2c_transform.T, ray_c)
                # intersect a ray with a scene in world axes
                intersection_points_w = []
                for obj in scene.objects:
                    intersection_points_w.extend(obj.find_all_intersections(ray_w, self.center))
                if len(intersection_points_w) > 0:
                    # find closest point of intersection
                    min_dist = np.inf
                    for point_w in intersection_points_w:
                        dist = self.eval_distance(point_w)
                        if dist < min_dist:
                            min_dist = dist
                    # set pixel value to corresponding value
                    if min_dist != np.inf:
                        rendered_scene[v, u, :] = self.eval_intensity(min_dist)
        return rendered_scene[::-1, ::-1, :]
