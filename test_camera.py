from synthetic import Camera
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.camera = Camera(500, 500, 480, 270, 960, 540, 0, 0.5, -10, 0, 0, 0)

    def test_eval_distance(self):
        point = np.array([
            [0.0, 0.5, -0.5]
        ]).T
        dist = self.camera.eval_distance(point)
        self.assertAlmostEqual(dist, 9.5)

if __name__ == '__main__':
    unittest.main()
