from turtle import pos
import cv2
import numpy as np
import tqdm
import random
from numpy.linalg import solve, qr
from utils import *
import matplotlib.pyplot as plt

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

class JacobianTools:

    @staticmethod
    def parse_jacobians(path):
        jacobians = {}
        for fname in os.listdir(path):
            if fname.endswith('txt'):
                with open(os.path.join(path, fname)) as f:
                    s = f.read()
                s = s.replace('cos', 'np.cos')
                s = s.replace('sin', 'np.sin')
                s = s.replace('^', '**')
                jacobians[fname.split('.')[0]] = s
        return jacobians

    @staticmethod
    def eval_jacobian_C(jacobians, col, row):
        # parameters: x, y, z, fx, fy, v_0, u_0, roll, pitch, yaw, t_x, t_y, t_z
        # dA1 (x-coordinate of pixel)
        if row % 2 == 0:
            # dA1 / dx
            if col % 3 == 0:
                formula = jacobians['dA1dx']
            # dA1 / dy
            elif col % 3 == 1:
                formula = jacobians['dA1dy']
            # dA1 / dz
            else:
                formula = jacobians['dA1dz']
        # dA2 (y-coordinate of pixel)
        else:
            # dA2 / dx
            if col % 3 == 0:
                formula = jacobians['dA2dx']
            # dA2 / dy
            elif col % 3 == 1:
                formula = jacobians['dA2dy']
            # dA2 / dz
            else:
                formula = jacobians['dA2dz']
        return formula

    @staticmethod
    def eval_jacobian_B(jacobians, col, row):
        # parameters: x, y, z, fx, fy, v_0, u_0, roll, pitch, yaw, t_x, t_y, t_z
        # dA1 (x-coordinate of pixel)
        if row % 2 == 0:
            # dA1 / droll
            if col % 6 == 0:
                formula = jacobians['dA1droll']
            # dA1 / dpitch
            elif col % 6 == 1:
                formula = jacobians['dA1dpitch']
            # dA1 / dyaw
            elif col % 6 == 2:
                formula = jacobians['dA1dyaw']
            # dA1 / dtx
            elif col % 6 == 3:
                formula = jacobians['dA1dtx']
            # dA1 / dty
            elif col % 6 == 4:
                formula = jacobians['dA1dty']
            # dA1 / dtz
            else:
                formula = jacobians['dA1dtz']
        # dA2 (y-coordinate of pixel)
        else:
            # dA1 / droll
            if col % 6 == 0:
                formula = jacobians['dA2droll']
            # dA1 / dpitch
            elif col % 6 == 1:
                formula = jacobians['dA2dpitch']
            # dA1 / dyaw
            elif col % 6 == 2:
                formula = jacobians['dA2dyaw']
            # dA1 / dtx
            elif col % 6 == 3:
                formula = jacobians['dA2dtx']
            # dA1 / dty
            elif col % 6 == 4:
                formula = jacobians['dA2dty']
            # dA1 / dtz
            else:
                formula = jacobians['dA2dtz']
        return formula

    @classmethod
    def eval_jacobian(cls, jacobians, projections_dict, points_dict, cameras_dict, show=False):
        '''
        Jacobian: matrix of partial derivatives
        rows: observations of state: 2 x N * points (2 rows per point, most of them not visible) + 2 x 6 * M cameras (2 rows per point, w.r.t. visible points)
        cols: partial derivatives variables: x1, y1, z1, x2, y2, z2, ..., 
        '''
        jacobian_mtx = np.zeros((len(projections_dict) * 2, len(points_dict) * 3 + 6 * len(cameras_dict)))

        for obs_id, obs in projections_dict.items():
            cam = cameras_dict[obs.cam_id]
            point3d = points_dict[obs.p_id]
            for row in range(obs_id * 2, obs_id * 2 + 2):
                # define col for sub-matrix C
                for col in range(obs.p_id * 3, obs.p_id * 3 + 3):
                    f = cls.eval_jacobian_C(jacobians, col, row)
                    f = f.replace('f_x', f'{cam.intrinsics.fx}')
                    f = f.replace('f_y', f'{cam.intrinsics.fy}')
                    f = f.replace('roll', f'{cam.roll}')
                    f = f.replace('pitch', f'{cam.pitch}')
                    f = f.replace('yaw', f'{cam.yaw}')
                    f = f.replace('t_x', f'{cam.center[0, 0]}')
                    f = f.replace('t_y', f'{cam.center[1, 0]}')
                    f = f.replace('t_z', f'{cam.center[2, 0]}')
                    f = f.replace('x', f'{point3d.x}')
                    f = f.replace('y', f'{point3d.y}')
                    f = f.replace('z', f'{point3d.z}')
                    if row % 2 == 0:
                        f = f.replace('u_0', f'{obs.x}')
                    else:
                        f = f.replace('v_0', f'{obs.y}')
                    jacobian_mtx[row, col] = eval(f)
                # define col for sub-matrix B
                for col in range(len(points_dict) * 3 + obs.cam_id * 6, len(points_dict) * 3 + obs.cam_id * 6 + 6):
                    f = cls.eval_jacobian_B(jacobians, col, row)
                    f = f.replace('f_x', f'{cam.intrinsics.fx}')
                    f = f.replace('f_y', f'{cam.intrinsics.fy}')
                    f = f.replace('roll', f'{cam.roll}')
                    f = f.replace('pitch', f'{cam.pitch}')
                    f = f.replace('yaw', f'{cam.yaw}')
                    f = f.replace('t_x', f'{cam.center[0, 0]}')
                    f = f.replace('t_y', f'{cam.center[1, 0]}')
                    f = f.replace('t_z', f'{cam.center[2, 0]}')
                    f = f.replace('x', f'{point3d.x}')
                    f = f.replace('y', f'{point3d.y}')
                    f = f.replace('z', f'{point3d.z}')
                    if row % 2 == 0:
                        f = f.replace('u_0', f'{obs.x}')
                    else:
                        f = f.replace('v_0', f'{obs.y}')

                    jacobian_mtx[row, col] = eval(f)
        

        def forceAspect(ax,aspect=1):
            im = ax.get_images()
            extent =  im[0].get_extent()
            ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(jacobian_mtx)
            forceAspect(ax)
            #ax.colorbar()
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(np.dot(jacobian_mtx.T, jacobian_mtx))
            forceAspect(ax)
            plt.show()


        return jacobian_mtx

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
            x = solve(A, -p0)
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

class Projection:

    def __init__(self, x, y, cam_id, p_id) -> None:
        self.x = x
        self.y = y
        self.cam_id = cam_id
        self.p_id = p_id

class Point3d:

    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

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
        a = random.random()
        b = random.random()
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

if __name__ == '__main__':
    w = 960
    h = 540
    intrinsics = Intrinsics(fx=500, fy=500, cx=w/2, cy=h/2, width=w, height=h)


    out = cv2.VideoWriter('rendered.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1.0, (w, h), True)
    # initialize scene
    scene = Scene()
    # initialize cameras and visualize rendered scene on each camera
    cameras = []
    # set up circle path
    # X-components of the camera's positions in 3D space
    xs = [(10.0 / 18.0) * x - 5.0 for x in range(18)]
    r = 4
    # Z-components of the camera's positions in 3D space
    zs = [-np.sqrt(r ** 2 - x ** 2) for x in xs]
    zs = [z if not np.isnan(z) else r for z in zs]
    for index, (x, z) in enumerate(zip(xs, zs)):
        position = (x, 0.5, z)
        orientation = (0, 0, 0) #-180 * np.arctan(z / r) / np.pi))
        cameras.append(Camera(intrinsics, position, orientation))

    data_sampler = DataSampler()
    projections_dict, points_dict, cameras_dict = data_sampler.sample(scene, cameras)

    # generate Jacobian matrix
    jacobians = JacobianTools.parse_jacobians('./ba_jacobian_1')
    JacobianTools.eval_jacobian(jacobians, projections_dict, points_dict, cameras_dict)

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
