import unittest
import fastfiz as ff
from fastfiz_env.utils.wrappers import deg_to_vec, vec_to_deg, vec_to_abs_deg
import numpy as np


class TestFeatures(unittest.TestCase):

    def test_deg_to_vec(self):
        self.assertTrue(np.allclose(deg_to_vec(0), [1, 0]))
        self.assertTrue(np.allclose(deg_to_vec(90), [0, 1]))
        self.assertTrue(np.allclose(deg_to_vec(180), [-1, 0]))
        self.assertTrue(np.allclose(deg_to_vec(270), [0, -1]))

    def test_vec_to_abs_deg(self):
        self.assertEqual(vec_to_abs_deg([1, 0]), 0)
        self.assertEqual(vec_to_abs_deg([0, 1]), 90)
        self.assertEqual(vec_to_abs_deg([-1, 0]), 180)
        self.assertEqual(vec_to_abs_deg([0, -1]), 270)

    def test_vec_to_deg(self):
        self.assertEqual(vec_to_deg([1, 0]), 0)
        self.assertEqual(vec_to_deg([0, 1]), 90)
        self.assertEqual(vec_to_deg([-1, 0]), 180)
        self.assertEqual(vec_to_deg([0, -1]), -90)


if __name__ == "__main__":
    unittest.main()
