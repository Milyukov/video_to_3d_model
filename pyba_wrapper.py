import camera
import numpy as np
import sampling
import scene

from pyba.CameraNetwork import CameraNetwork
from pyba.pyba import bundle_adjust 

def prepare_data(cameras, projections_list, n_points):
    """
    
    """
    assert len(cameras) > 0

    # points2d: numpy array of shape CxTxJx2, where C is the number of cameras, J is the number of joints and T is the time axis
    points2d = np.zeros((1, len(cameras), n_points, 2))
    for proj in projections_list:
        points2d[0, proj.cam_id, proj.p_id, 0] = proj.u
        points2d[0, proj.cam_id, proj.p_id, 1] = proj.v
    # calib is a nested dictionary where keys are camera id's, indexed starting from 0 up to n_cameras-1
    fu = cameras[0].intrinsics.fu
    fv = cameras[0].intrinsics.fv
    cu = cameras[0].intrinsics.cu
    cv = cameras[0].intrinsics.cv
    calib = {
        0: {
            'R': cameras[0].w2c_transform,
            'tvec': cameras[0].center[:, 0],
            'intr': np.array([[fu, 0, cu],
                              [0, fv, cv],
                              [0, 0, 1]]),
            'distort': np.array([0., 0., 0., 0., 0.])
        },
        'points2d': points2d
    }
    return calib

if __name__ == '__main__':
    w = 960
    h = 540
    intrinsics = camera.Intrinsics(fu=500, fv=500, cu=w/2, cv=h/2, width=w, height=h)

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

    # perfrom reference BA
    d = prepare_data(cameras, projections_list, len(points_list))
    camNet = CameraNetwork(points2d=d['points2d'], calib=d)
    bundle_adjust(camNet)
