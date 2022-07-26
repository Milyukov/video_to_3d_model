class Camera:

    def __init__(self) -> None:
        pass

    def project_points(self):
        pass

class Model3D:

    def __init__(self) -> None:
        pass

class Scene:

    def __init__(self) -> None:
        pass

class FeatureTracker:

    def __init__(self) -> None:
        pass

    def track(self, points):
        pass

if __name__ == '__main__':
    # initialize scene
    scene = Scene()
    # initialize cameras
    cameras = []
    for cam_num in range(10):
        cameras.append(Camera())
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
