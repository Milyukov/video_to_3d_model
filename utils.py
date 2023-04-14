import os
import cv2
import numpy as np
#from synthetic import Box

def px_coord_xz(coord):
    return (np.int((-coord[0] + 5.0) * 100), np.int((-coord[2] + 5.0) * 100))

def draw_xz_projection_for_a_plane(plane, image, origin, color, ray=None):
    """_summary_
    Renders projection of a plane given it's parameters and virtual camera defined by origin and ray
    Args:
        plane (Plane): plane parametrization in 3D space
        image (ndarray): array used as a matrix for a projection
        origin (ndarray): principal point of a virtual camera
        color (tuple): color for lines visuzalization
        ray (ndarray, optional): Direction of a camera's view. Defaults to None.
    """
    points = np.concatenate((plane.p0, plane.points), axis=1).T
    for p_id in range(len(points)):
        p = points[p_id:p_id+1, :].T
        if ray is None:
            ray = p - origin
        intersection = Box.find_intersection(ray, origin, plane, True)

        if intersection is not None:
            # draw a ray
            origin_xz = px_coord_xz(origin)
            intersection_xz = px_coord_xz(intersection)
            cv2.line(image, origin_xz, intersection_xz, (0, 0, 255))
        else:
            origin_xz = px_coord_xz(origin)
            intersection_xz = px_coord_xz(origin + ray * 1000)
            cv2.line(image, origin_xz, intersection_xz, (0, 0, 255))
        # draw plane projection
        prev_point = None
        first_point = None
        for p_id, p in enumerate(plane.points.T):
            p_xz = px_coord_xz(p)
            color_val = (p_id + 1) * 64
            cv2.circle(image, p_xz, 7, (color_val, color_val, color_val), 3)
            if first_point is None:
                first_point = p_xz
            if prev_point is not None:
                cv2.line(image, prev_point, p_xz, color)
            prev_point = p_xz
        cv2.line(image, prev_point, first_point, color)
        cv2.imshow("", image)
        cv2.waitKey(0)

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
