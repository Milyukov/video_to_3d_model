from synthetic import Camera, Intrinsics, Scene, Box
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import unittest

def make_grid():
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

        cam_yaw_pos = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, 0, 30))
        cam_yaw_neg = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, 0, -30))
        cam_pitch_pos = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, 30, 0))
        cam_pitch_neg = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (0, -30, 0))
        cam_roll_pos = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (30, 0, 0))
        cam_roll_neg = Camera(Intrinsics(500, 500, 480, 270, 960, 540), (0, 0.5, -10) , (-30, 0, 0))

        self.cases_cameras = [
            cam_roll_neg, self.default_camera, cam_roll_pos,
            cam_pitch_neg, self.default_camera, cam_pitch_pos,
            cam_yaw_neg, self.default_camera, cam_yaw_pos
            ]
        
        scene_grid = make_grid()
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
                     width {camera.intrinsics.width}, height: {camera.intrinsics.height}
                     fx: {camera.intrinsics.K[0, 0]}, fy: {camera.intrinsics.K[1, 1]}
                     cx: {camera.intrinsics.K[0, 2]}, cy: {camera.intrinsics.K[1, 2]}
                     tx: {camera.center[0, 0]}, ty: {camera.center[1, 0]}, tz: {camera.center[2, 0]}
                     roll: {camera.roll}, pitch: {camera.pitch}, yaw: {camera.yaw}
                     ''')
            plt.savefig(fig_path / f'render_{i}.png')
            plt.clf()
            
        





if __name__ == '__main__':
    
    unittest.main()
    


