import cv2
import numpy as np
from synthetic import Box

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
