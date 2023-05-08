import numpy as np

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
            'tvec': cameras[0].center,
            'intr': np.array([[fu, 0, cu],
                              [0, fv, cv],
                              [0, 0, 1]]),
            'distort': np.array([0., 0., 0., 0., 0.])
        },
        'points2d': points2d
    }
    return calib
