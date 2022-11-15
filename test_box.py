from synthetic import Box, Plane
import numpy as np
import unittest

from utils import *

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.box = Box(0, 0.5, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        
    def test_find_intersection(self):
        ray = np.array([
            [0.0, 0.0, 1.0]
        ]).T
        origin = np.array([
            [0, 0.5, -10]
        ]).T
        intersection = Box.find_intersection(ray, origin, self.box.front_plane)
        self.assertAlmostEqual(intersection[0, 0], 0.0)
        self.assertAlmostEqual(intersection[1, 0], 0.5)
        self.assertAlmostEqual(intersection[2, 0], -0.5)

        intersection = Box.find_intersection(ray, origin, self.box.rare_plane)
        self.assertAlmostEqual(intersection[0, 0], 0.0)
        self.assertAlmostEqual(intersection[1, 0], 0.5)
        self.assertAlmostEqual(intersection[2, 0], 0.5)

        ray = np.array([
            [-1.0, -2.0, 9.5]
        ]).T
        intersection = Box.find_intersection(ray, origin, self.box.front_plane)
        self.assertTrue(intersection is None)


if __name__ == '__main__':
    unittest.main()

