import os
import numpy as np
import matplotlib.pyplot as plt

class JacobianTools:

    @staticmethod
    def parse_jacobians(path):
        """
        Parses analytical jacobians from *.txt files in path.
        Args:
            path (str): path to *.txt files with jacobians.

        Returns:
            dict: parsed jacobians, key is a specific partial derivative.
        """
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
    def eval_jacobian_C(jacobians, row, col):
        """
        Returns formula for given row, col of jacobian matrix 
        (left part of the matrix with 3D coordinates derivatives)
        Args:
            jacobians (dict): parsed jacobian formulas.
            row (int): row index of jacobian matrix (starting from 0)
            col (int): col index of jacobian matrix (starting from 0)

        Returns:
            str: formula for given row, col of jacobian matrix
        """
        # parameters: x, y, z, fu, fv, v_0, u_0, roll, pitch, yaw, t_x, t_y, t_z
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
    def eval_jacobian_B(jacobians, row, col):
        """
        Returns formula for given row, col of jacobian matrix 
        (right part of the matrix with cameras position/orientation derivatives)
        Args:
            jacobians (dict): parsed jacobian formulas.
            row (int): row index of jacobian matrix (starting from 0)
            col (int): col index of jacobian matrix (starting from 0)

        Returns:
            str: formula for given row, col of jacobian matrix
        """
        # parameters: x, y, z, fu, fv, v_0, u_0, roll, pitch, yaw, t_x, t_y, t_z
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
    def eval_jacobian(cls, jacobians, projections_list, points_list, cameras_list, show=False):
        """
        Generates Jacobian: matrix of partial derivatives for given observations and estimations.
        Rows: observations of state: 2 x N * points (2 rows per point, most of them not visible) + 2 x 6 * M cameras (2 rows per point, w.r.t. visible points)
        Cols: partial derivatives variables: x1, y1, z1, x2, y2, z2, ..., tx1, tx2, tx3, roll1, pitch1, yaw1, ...
        Args:
            jacobians (dict): parsed jacobian formulas
            projections_list (list): projections of 3D points for each camera and each 3D point.
            points_list (list of Point3d): estimated 3D points.
            cameras_list (list): cameras used for observations.
            show (bool, optional): Flag signaling whether to show Jacobian matrix structure. Defaults to False.

        Returns:
            ndarray: jacobian matrix.
        """
        jacobian_mtx = np.zeros((len(projections_list) * 2, len(points_list) * 3 + 6 * len(cameras_list)))

        for obs_id, obs in enumerate(projections_list):
            cam = cameras_list[obs.cam_id]
            point3d = points_list[obs.p_id]
            for row in range(obs_id * 2, obs_id * 2 + 2):
                # define col for sub-matrix C
                for col in range(obs.p_id * 3, obs.p_id * 3 + 3):
                    f = cls.eval_jacobian_C(jacobians, col, row)
                    f = f.replace('u_0', f'{cam.intrinsics.cu}')
                    f = f.replace('v_0', f'{cam.intrinsics.cv}')
                    f = f.replace('f_u', f'{cam.intrinsics.fu}')
                    f = f.replace('f_v', f'{cam.intrinsics.fv}')
                    f = f.replace('roll', f'{cam.roll}')
                    f = f.replace('pitch', f'{cam.pitch}')
                    f = f.replace('yaw', f'{cam.yaw}')
                    f = f.replace('t_x', f'{cam.center[0, 0]}')
                    f = f.replace('t_y', f'{cam.center[1, 0]}')
                    f = f.replace('t_z', f'{cam.center[2, 0]}')
                    f = f.replace('x', f'{point3d.x}')
                    f = f.replace('y', f'{point3d.y}')
                    f = f.replace('z', f'{point3d.z}')
                    jacobian_mtx[row, col] = eval(f)
                # define col for sub-matrix B
                for col in range(len(points_list) * 3 + obs.cam_id * 6, len(points_list) * 3 + obs.cam_id * 6 + 6):
                    f = cls.eval_jacobian_B(jacobians, col, row)
                    f = f.replace('u_0', f'{cam.intrinsics.cu}')
                    f = f.replace('v_0', f'{cam.intrinsics.cv}')
                    f = f.replace('f_u', f'{cam.intrinsics.fu}')
                    f = f.replace('f_v', f'{cam.intrinsics.fv}')
                    f = f.replace('roll', f'{cam.roll}')
                    f = f.replace('pitch', f'{cam.pitch}')
                    f = f.replace('yaw', f'{cam.yaw}')
                    f = f.replace('t_x', f'{cam.center[0, 0]}')
                    f = f.replace('t_y', f'{cam.center[1, 0]}')
                    f = f.replace('t_z', f'{cam.center[2, 0]}')
                    f = f.replace('x', f'{point3d.x}')
                    f = f.replace('y', f'{point3d.y}')
                    f = f.replace('z', f'{point3d.z}')

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
