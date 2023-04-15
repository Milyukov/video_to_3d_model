from synthetic import Camera, Intrinsics, Scene, Box
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import unittest

def make_grid_scene():
    '''
    Makes a scene with 3x3 structure of 9 boxes aligned in XY plane
    '''
    scene_grid = Scene()
    scene_grid.objects=[]
    scene_grid.objects.append(Box(-3, 3, 0, 1, 1, 1, 0, 0, 0))
    scene_grid.objects.append(Box(0, 3, 0, 1, 1, 1, 0, 0, 0))
    scene_grid.objects.append(Box(3, 3, 0, 1, 1, 1, 0, 0, 0))

    scene_grid.objects.append(Box(-3, 0, 0, 1, 1, 1, 0, 0, 0))
    scene_grid.objects.append(Box(0, 0, 0, 1, 1, 1, 0, 0, 0))
    scene_grid.objects.append(Box(3, 0, 0, 1, 1, 1, 0, 0, 0))

    scene_grid.objects.append(Box(-3, -3, 0, 1, 1, 1, 0, 0, 0))
    scene_grid.objects.append(Box(0, -3, 0, 1, 1, 1, 0, 0, 0))
    scene_grid.objects.append(Box(3, -3, 0, 1, 1, 1, 0, 0, 0))
    return scene_grid

def make_axes_scene():
    '''
    Makes a scene with 3 orthogonal axes - X, Y, Z
    '''
    scene_grid = scene.Scene()
    scene_grid.objects=[]
    scene_grid.objects.append(Box(3, 0, 0, 6, 0.1, 0.1, 0, 0, 0 ))
    scene_grid.objects.append(Box(0, 0, 3, 6, 0.1, 0.1, 0, 0, -np.deg2rad(90) ))
    scene_grid.objects.append(Box(0, 3, 0, 6, 0.1, 0.1, np.deg2rad(90), 0, 0 ))
    return scene_grid

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.camera = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, 0, 0))

    def test_eval_distance(self):
        point = np.array([
            [0.0, 0.5, -0.5]
        ]).T
        dist = self.camera.eval_distance(point)
        self.assertAlmostEqual(dist, 9.5)

class TestCameraParameters(unittest.TestCase):

    def setUp(self) -> None:
        self.default_camera = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, 0, 0))
        self.default_scene = Scene()

        #camera set to test simple single-axis rotations
        cam_yaw_pos = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, 0, np.deg2rad(30)))
        cam_yaw_neg = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, 0, -np.deg2rad(30)))
        cam_pitch_pos = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, np.deg2rad(30), 0))
        cam_pitch_neg = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, -np.deg2rad(30), 0))
        cam_roll_pos = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (np.deg2rad(30), 0, 0))
        cam_roll_neg = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (-np.deg2rad(30), 0, 0))
        
        #camera set 1 to test complex multiaxis rotations
        cam_roll_pitch = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (np.deg2rad(30), np.deg2rad(30), 0))
        cam_roll_yaw  = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (np.deg2rad(30), 0, np.deg2rad(30)))
        cam_pitch_yaw = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, np.deg2rad(30), np.deg2rad(30)))
        
        #camera set 2 to test complex multiaxis sequential rotations
        axes_camera_init= Camera(Intrinsics(500, 500, 480, 270, 960, 540), (3, 2, -5), (0, 0 ,0))
        axes_camera_roll = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (3, 2, -5), (-np.pi / 2, 0 , 0))
        axes_camera_roll_pitch = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (3, 2, -5), (-np.pi / 2, np.pi / 6 , 0))
        axes_camera_roll_pitch_yaw = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (3, 2, -5), (-np.pi / 2, np.pi / 6 , - np.pi / 6))
        self.cases_cameras = [
            axes_camera_init, axes_camera_roll,  axes_camera_roll_pitch, axes_camera_roll_pitch_yaw 
            ]
        
        scene_grid = make_axes_scene()
        self.cases_scenes = [scene_grid] * len(self.cases_cameras)
        return super().setUp()
    
    def test_render_scenes_cameras(self):
        '''
        Renders the image for (camera, scene) pairs provided in 
        self.cases_cameras and self.cases_scenes. Lengths of these lists must be equal.
        If one or both of these lists are empty, default scene or camera are appended to them.
        Images are stored in test_results_data folder.
        '''
        if len(self.cases_scenes) == 0:
            self.cases_scenes.append(self.default_scene)
        if len(self.cases_cameras) == 0:
            self.cases_cameras.append(self.default_camera)
        if len(self.cases_cameras) != len(self.cases_scenes):
            raise Exception("""Cameras and scenes lists must be of equal size""")

        for i, case in enumerate(zip(self.cases_cameras, self.cases_scenes)):
            camera, scene = case
            fig_path = Path('test_results_data')
            plt.imshow(camera.render_scene(scene))
            plt.text(0, 0, 
                     f'''
                     width {camera.intrinsics.width:.2f}, height: {camera.intrinsics.height:.2f}
                     fu: {camera.intrinsics.fu:.2f}, fv: {camera.intrinsics.fv:.2f}
                     cu: {camera.intrinsics.cu:.2f}, cv: {camera.intrinsics.cv:.2f}
                     tx: {camera.center[0, 0]:.2f}, ty: {camera.center[1, 0]:.2f}, tz: {camera.center[2, 0]:.2f}
                     roll: {np.rad2deg(camera.roll):.2f}, pitch: {np.rad2deg(camera.pitch):.2f}, yaw: {np.rad2deg(camera.yaw):.2f}
                     view axis: {camera.w2c_transform[2, 0]:.2f}, {camera.w2c_transform[2, 1]:.2f}, {camera.w2c_transform[2, 2]:.2f}
                     ''')
            plt.savefig(fig_path / f'render_{i}.png')
            plt.clf()
            
        





if __name__ == '__main__':
    
    unittest.main()
    


