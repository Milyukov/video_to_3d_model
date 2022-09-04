import cv2
import numpy as np
import tqdm
from numpy.linalg import solve, qr

def get_rotation_matrix(roll, pitch, yaw):
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    # roll
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0.0],
        [np.sin(roll_rad), np.cos(roll_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    # yaw
    R_yaw = np.array([
        [np.cos(yaw_rad), 0.0, -np.sin(yaw_rad)],
        [0.0, 1.0, 0.0],
        [np.sin(yaw_rad), 0.0, np.cos(yaw_rad)]  
    ])
    # pitch
    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0.0, np.sin(pitch_rad), np.cos(pitch_rad)]  
    ])

    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    return R

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
        self.R = get_rotation_matrix(roll, pitch, yaw)
        # maximum distance of visibility in meters
        self.max_distance = 40

    def project_points(self):
        pass
    
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

class Model3D:

    def __init__(self) -> None:
        pass

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
        self.left_plane = Plane(self.points[:, [0, 3, 4, 7]], np.cross(self.points[:, 0] - self.points[:, 4], self.points[:, 0] - self.points[:, 3]))
        self.right_plane = Plane(self.points[:, [1, 2, 5, 6]], np.cross(self.points[:, 1] - self.points[:, 2], self.points[:, 1] - self.points[:, 5]))
        self.front_plane = Plane(self.points[:, [2, 3, 7, 6]], np.cross(self.points[:, 2] - self.points[:, 3], self.points[:, 2] - self.points[:, 6]))
        self.rare_plane = Plane(self.points[:, [0, 1, 5, 4 ]], np.cross(self.points[:, 0] - self.points[:, 1], self.points[:, 0] - self.points[:, 4]))

    @staticmethod
    def find_intersection(ray, l0, plane, debug=False):
        # plane is parametrized as: ax + by + cz + d = 0
        # or rather it is parametrized as a point P0 and a normal n
        # ray is parametrized as starting point l0 and direction l
        # shift along the ray to point of intersection is t (P = l0 + t * l)
        # t = (P0 - l0) * n / (l * n)
        # P = l0 + t * l
        denom = np.dot(ray.T, plane.normal)
        if denom != 0:
            t = np.dot((plane.p0 - l0).T, plane.normal) / denom
            point = l0 + t * ray

            # check if projection of point is inside the borders of the plane
            p0 = plane.points[:, 0:1]
            p1 = plane.points[:, 1:2]
            p2 = plane.points[:, 2:3]
            p3 = plane.points[:, 3:4]

            r = point - p0
            r1 = p1 - p0
            r2 = p3 - p0

            A = np.concatenate((r1, r2), axis=1)
            Q, R = qr(A)
            b = np.dot(Q.T, r)
            #if np.all(b[2, :] == 0) and np.all(R[2, :] == 0):
            # R = R[:2, :]
            # b = b[:2]
            x = solve(R, b)

            if 0 <= x[0] <= 1:
                if 0 <= x[1] <= 1:
                    return point

        return point if debug else None

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

class Scene:

    def __init__(self) -> None:
        """Initialize scene as a number of geometric promitives in 3D
        """
        self.objects = []
        self.objects.append(Box(0, 0.5, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 45.0))
        self.objects.append(Box(0, 1.25, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0))
        self.objects.append(Box(1.5, 0.5, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 30.0))

class FeatureTracker:

    def __init__(self) -> None:
        pass

    def track(self, points):
        pass

if __name__ == '__main__':
    w = 960
    h = 540
    intrinsics = Intrinsics(fx=500, fy=500, cx=w/2, cy=h/2, width=w, height=h)


    out = cv2.VideoWriter('rendered.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1.0, (w, h), True)
    # initialize scene
    scene = Scene()
    # initialize cameras and visualize rendered scene on each camera
    cameras = []
    xs = [(10.0 / 18.0) * x - 5.0 for x in range(18)]
    r = 4
    zs = [-np.sqrt(r ** 2 - x ** 2) for x in xs]
    zs = [z if not np.isnan(z) else r for z in zs]
    for index, (x, z) in enumerate(zip(xs, zs)):
        position = (x, 0.5, z)
        orientation = (0, 0, 0) #-180 * np.arctan(z / r) / np.pi))
        cameras.append(Camera(intrinsics, position, orientation))
    frames = []

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 50)
    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 2

    for cam_id, cam in enumerate(cameras):
        frame = cam.render_scene(scene)
        frame = cv2.putText(np.array(frame), f'Frame #{cam_id}', (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(np.array(frame), f'Cam pos: {cam.center}', (50, 100), font, fontScale, color, thickness, cv2.LINE_AA)
        frames.append(frame)
        out.write(frame)
    out.release()
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
