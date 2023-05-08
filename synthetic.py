import scene
import camera
import jacobian_tools
import sampling

import cv2
import numpy as np
from utils import *

from pyba.CameraNetwork import CameraNetwork
from pyba_wrapper import prepare_data
from pyba.pyba import bundle_adjust 

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
    # X-components of the camera's positions in 3D space
    xs = [(10.0 / 18.0) * x - 5.0 for x in range(18)]
    r = 4
    # Z-components of the camera's positions in 3D space
    zs = [-np.sqrt(r ** 2 - x ** 2) for x in xs]
    zs = [z if not np.isnan(z) else r for z in zs]
    for index, (x, z) in enumerate(zip(xs, zs)):
        position = (x, 0.5, z)
        orientation = (0, 0, 0) #-180 * np.arctan(z / r) / np.pi))
        cameras.append(camera.Camera(intrinsics, position, orientation))

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
