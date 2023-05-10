import scene
import camera
import jacobian_tools
import sampling

import cv2
import numpy as np
from utils import *

if __name__ == '__main__':
    w = 960
    h = 540
    intrinsics = camera.Intrinsics(fu=500, fv=500, cu=w/2, cv=h/2, width=w, height=h)


    out = cv2.VideoWriter('./test_results_data/rendered.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1.0, (w, h), True)
    # initialize scene
    synthetic_scene = scene.Scene()
    # initialize cameras and visualize rendered scene on each camera
    cameras = []
    # set up circle path
    r = 4
    # Z-components of the camera's positions in 3D space
    zs = [(z / 18.0) * 2 * r - r for z in range(19)]
    # x**2 + z**2 = r**2
    # x = +-sqrt(r**2 - z**2)
    # X-components of the camera's positions in 3D space
    xs = [-np.sqrt(r ** 2 - z ** 2) for z in zs]
    xs = [x if not np.isnan(x) else r for x in xs]
    # visualize on 2D top-view projection X-Z
    scale = 30
    # 10 m x 10 m
    projection_x_z = np.zeros((10 * scale, 10 * scale))
    for (x, z) in zip(xs, zs):
        position = (x, 0.5, z)
        '''

            ^ X
            |   /|
            |  /yaw
            | /  |
            |/_ _|_ _ _ _ _> z

        '''
        if np.abs(x) > 10e-3:
            yaw = np.sign(x) * np.arctan(z / x)
        else:
            yaw = np.sign(z) * np.pi / 2
        orientation = 0, 0, -yaw
        print(position)
        print(yaw)

        center_coords = np.array([-np.int(x * scale) + 5 * scale, np.int(z * scale) + 5 * scale])
        cv2.circle(projection_x_z, center_coords, 5, (255), 1)
        ray = np.array([r * scale * np.cos(yaw), r * scale * np.sin(yaw)])
        ray_end = center_coords + ray
        cv2.line(projection_x_z, center_coords.astype(np.int), ray_end.astype(np.int), (255), 1)

        cameras.append(camera.Camera(intrinsics, position, orientation))

    cv2.imshow("", projection_x_z)
    cv2.waitKey()

    data_sampler = sampling.DataSampler()
    projections_list, points_list = data_sampler.sample(synthetic_scene, cameras)

    # generate Jacobian matrix
    jacobians = jacobian_tools.JacobianTools.parse_jacobians('./ba_jacobian_int')
    jacobian_tools.JacobianTools.eval_jacobian(jacobians, projections_list, points_list, cameras)

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
        frame = cam.render_scene(synthetic_scene)
        cv2.imwrite(f"./test_results_data/camera_0_img_00000{cam_id}.jpg", frame)
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
    tracker = sampling.FeatureTracker()
    tracker.track(points)
    # run orientation estimation and compare with true orientations
    # run 3D reconstruction and compare with scene
    # estimate point cloud with semantic and instance infor from the ground truth scene
    # estimate 3D model and compare with ground truth model
