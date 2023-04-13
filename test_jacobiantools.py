import jacobian_tools
import camera
import scene_objects
import scene 
import sampling
from utils import get_rotation_matrix

import unittest
import numpy as np
import copy
import pandas as pd

def make_parameters_dict(point, camera):
    """
    Make a dict with parameters necessary for jackobian evaluation
    from point and camera
    """
    jackobian_parameters = {
            'f_x': camera.intrinsics.fx, 
            'f_y': camera.intrinsics.fy,
            'u_0': camera.intrinsics.cx,
            'v_0': camera.intrinsics.cy,
            'roll': camera.roll,
            'pitch': camera.pitch,
            'yaw': camera.yaw,
            't_x': camera.center[0, 0],
            't_y': camera.center[1, 0],
            't_z': camera.center[2, 0],
            'x': point.x,
            'y': point.y,
            'z': point.z,
            'np': np
            }
    return jackobian_parameters

def generate_circular_movement():
    """
    Sets up circle path for camera,
    returns tuple of lists: (positions, orientations)
    """
    positions = []
    orientations = []
    # X-components of the camera's positions in 3D space
    xs = [(10.0 / 18.0) * x - 5.0 for x in range(5)]
    r = 4
    # Z-components of the camera's positions in 3D space
    zs = [-np.sqrt(r ** 2 - x ** 2) for x in xs]
    zs = [z if not np.isnan(z) else r for z in zs]
    for _, (x, z) in enumerate(zip(xs, zs)):
        positions.append((x, 0.5, z))
        orientations.append((0, 0, -180 * np.arctan(z / r) / np.pi))
    return positions, orientations

def generate_sample_data(intrinsics, positions, orientations):
    """
    Generates state and projections values for cameras 
    with given intrinsics, positions and orientations
    """
    # initialize scene
    sampled_scene = scene.Scene()
    # initialize cameras and visualize rendered scene on each camera
    cameras = []
    
    for position, orientation in zip(positions, orientations):
        cameras.append(camera.Camera(intrinsics, position, orientation))

    data_sampler = sampling.DataSampler()
    projections_dict, points_dict, cameras_dict = data_sampler.sample(sampled_scene, cameras)
    return projections_dict, points_dict, cameras_dict

def get_variated_projection(point, camera, variations):
    """
    Generates projection for given camera and point with variated parameters
    """    
    parameters_change = dict.fromkeys(
        ['point.x', 'point.y', 'point.z', 
         'camera.roll', 'camera.pitch', 'camera.yaw', 
         'camera.t_x', 'camera.t_y', 'camera.t_z'], 0)
    for parameter, variation in variations.items():
        parameters_change[parameter] += variation

    point.x += parameters_change['point.x']
    point.y += parameters_change['point.y']
    point.z += parameters_change['point.z']

    camera.roll += np.rad2deg(parameters_change['camera.roll'])
    camera.pitch += np.rad2deg(parameters_change['camera.pitch'])
    camera.yaw += np.rad2deg(parameters_change['camera.yaw'])
    camera.R = get_rotation_matrix(camera.roll, camera.pitch, camera.yaw)

    camera.center[0, 0] += parameters_change['camera.t_x']
    camera.center[1, 0] += parameters_change['camera.t_y']
    camera.center[2, 0] += parameters_change['camera.t_z']

    return camera.project_point(np.array([
        [point.x], [point.y], [point.z]
    ]))



class TestJacobianIsFull(unittest.TestCase):

    def setUp(self) -> None:
        self.test_path = 'ba_jacobian_1'
        return super().setUp()

    def test_all_elements_present(self):
        """
        Test for all elements of a Jackobian are present
        """
        jackobian_elements = set(
            ['dA1dx', 'dA1dtx', 'dA1dpitch', 'dA1droll', 'dA1dty', 'dA1dtz', 'dA1dy', 'dA1dyaw', 'dA1dz', 
             'dA2dx', 'dA2dtx', 'dA2dpitch', 'dA2droll', 'dA2dty', 'dA2dtz', 'dA2dy', 'dA2dyaw', 'dA2dz']
            )
        expressions = set(jacobian_tools.JacobianTools.parse_jacobians(self.test_path).keys())
        self.assertTrue(
            jackobian_elements.issubset(expressions),
            msg=f"""Missing elements: {expressions.difference(jackobian_elements)}""")

class TestJackobianIsValid(unittest.TestCase):

    def setUp(self) -> None:
        self.test_path = 'ba_jacobian_1'
        self.expressions = jacobian_tools.JacobianTools.parse_jacobians(self.test_path)
        #set const variations for parameters
        self.const_variations = {
            'point.x': 0.1, 'point.y': 0.1, 'point.z': 0.1,
            'camera.roll': np.deg2rad(2.5), 'camera.pitch': np.deg2rad(2.5), 'camera.yaw': np.deg2rad(2.5),
            'camera.t_x': 0.1, 'camera.t_y': 0.1, 'camera.t_z': 0.1
            }
        self.variaton_mult = 1
        self.ju_idxs = {
                            'point.x': (0, 0), 'point.y':(0, 1), 'point.z': (0, 2),
                            'camera.roll': (0, 0), 'camera.pitch': (0, 1), 'camera.yaw': (0, 2),
                            'camera.t_x': (0, 3), 'camera.t_y': (0, 4), 'camera.t_z': (0, 5)
                        }
        self.jv_idxs = {
                            'point.x': (1, 0), 'point.y':(1, 1), 'point.z': (1, 2),
                            'camera.roll': (1, 0), 'camera.pitch': (1, 1), 'camera.yaw': (1, 2),
                            'camera.t_x': (1, 3), 'camera.t_y': (1, 4), 'camera.t_z': (1, 5)
                        }
        self.approx_delta = 1
        self.approx_test_result_columns = [
            'proj_id', 'variated_parameter', 'direction',
            'point.x', 'point.y', 'point.z', 
            'camera.roll', 'camera.pitch', 'camera.yaw',
            'camera.t_x', 'camera.t_y', 'camera.t_z',
            'u_nominal', 'v_nominal', 'u_var', 'v_var', 
            'resid_u', 'resid_v',
            'jcoeffs_u', 'jcoeffs_v', 'approx_dist'
            ]
        self.approx_test_results_list = []
        return super().setUp()
    
    def test_parsable_expressions(self):
        """
        Test if Jacobian element strings are valid for eval()
        """
    
        #make dummy parameters from dummy projection, point and camera
        jacobian_parameters = make_parameters_dict(
            scene_objects.Point3d(0, 0, 0),
            camera.Camera(
                camera.Intrinsics(fx=500, fy=500, cx=480, cy=270, width=960, height=540), 
                (0, 0.5, 10), (0, 0, 0)))

        for expression in self.expressions.values():
            with self.subTest(expression = expression):
                eval(expression, jacobian_parameters)

    def test_jacobian_approx(self):
        """     
        Test for Jacobian approximating camera model correctly
        """
        #generate nominal parameter values for system
        projections_dict, points_dict, cameras_dict = generate_sample_data(
            camera.Intrinsics(fx=500, fy=500, cx=480, cy=270, width=960, height=540), 
            [(0, 0.5, 10)],
            [(0, 0, 0)])
        
        for proj_id, nominal_projection in projections_dict.items():
            with self.subTest(proj_id = proj_id, nominal_projection=nominal_projection):
                nominal_point = points_dict[nominal_projection.p_id]
                nominal_camera = cameras_dict[nominal_projection.cam_id]
                
                #variate nominal camera and point parameters 
                #one by one by positive and negative amount
                for parameter, variation in self.const_variations.items():
                    with self.subTest(parameter=parameter, variation = self.variaton_mult * variation):
                        
                        #evaluating Jacobian element for current parameter
                        jacobian_parameters = make_parameters_dict(nominal_point, nominal_camera)
                        
                        if parameter in ['point.x', 'point.y', 'point.z']:
                            jcoeffs = np.array([
                                [eval(jacobian_tools.JacobianTools.eval_jacobian_C(
                                    self.expressions, self.ju_idxs[parameter][1], self.ju_idxs[parameter][0]), jacobian_parameters)],
                                [eval(jacobian_tools.JacobianTools.eval_jacobian_C(
                                self.expressions, self.jv_idxs[parameter][1], self.jv_idxs[parameter][0]), jacobian_parameters)]
                                ])
                        else:
                            jcoeffs = np.array([
                                [eval(jacobian_tools.JacobianTools.eval_jacobian_B(
                                    self.expressions, self.ju_idxs[parameter][1], self.ju_idxs[parameter][0]), jacobian_parameters)],
                                [eval(jacobian_tools.JacobianTools.eval_jacobian_B(
                                    self.expressions, self.jv_idxs[parameter][1], self.jv_idxs[parameter][0]), jacobian_parameters)]
                                ])

                        #make projections with a parameter variated in both directions 
                        for direction in [-1, 1]:
                            with self.subTest(direction = direction):
                                variated_projection = get_variated_projection(
                                    copy.deepcopy(nominal_point), 
                                    copy.deepcopy(nominal_camera), 
                                    {parameter: direction * variation})
                                #check L2 distance between residuals and its jackobian approximation
                        
                                #nominal projection might fit the image 
                                #but variated projection still might be off the image
                                if variated_projection[0, 0] is not None and variated_projection[1, 0] is not None:
                                    residual = np.array([
                                        [variated_projection[0, 0] - nominal_projection.x],
                                        [variated_projection[1, 0] - nominal_projection.y]
                                    ])
                                    #record the results 
                                    self.approx_test_results_list.append([
                                        proj_id, parameter, direction,
                                        nominal_point.x, nominal_point.y, nominal_point.z,
                                        nominal_camera.roll, nominal_camera.pitch, nominal_camera.yaw,
                                        nominal_camera.center[0, 0], nominal_camera.center[1, 0], nominal_camera.center[2, 0],
                                        nominal_projection.x, nominal_projection.y,
                                        variated_projection[0, 0], variated_projection[1, 0],
                                        residual[0, 0], residual[1, 0],
                                        jcoeffs[0, 0], jcoeffs[1, 0],  
                                        np.linalg.norm(residual - jcoeffs * direction * variation)
                                        ])
                            
                                    
                                    #test if approximation is close enough
                                    self.assertAlmostEqual(
                                        np.linalg.norm(residual - jcoeffs * direction * variation), 0.0,
                                        delta=self.approx_delta)
                    
    def tearDown(self) -> None:
        self.approx_test_results = pd.DataFrame(
            np.array(self.approx_test_results_list), columns=self.approx_test_result_columns)
        self.approx_test_results.to_csv('test_results_data/approx_test_results.csv')
        self.approx_test_results.to_excel('test_results_data/approx_test_results.xlsx')
        return super().tearDown()            
                


if __name__ == '__main__':
    unittest.main()