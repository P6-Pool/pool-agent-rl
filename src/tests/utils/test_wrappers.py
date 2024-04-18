import unittest
from fastfiz_env.wrappers.utils import cart2sph, sph2deg


class TestFeatures(unittest.TestCase):
    def test_cart2sph(self):
        x, y, z = 1, 1, 1
        az, el, r = cart2sph(x, y, z)
        self.assertAlmostEqual(az, 0.7853981633974483)
        self.assertAlmostEqual(el, 0.6154797086703873)
        self.assertAlmostEqual(r, 1.7320508075688772)

    def test_sph2deg(self):
        az, el, r = 0.7853981633974483, 0.6154797086703873, 1.7320508075688772
        phi, theta, r = sph2deg(az, el, r)
        self.assertAlmostEqual(phi, 45.0)
        self.assertAlmostEqual(theta, 35.26438968275466)
        self.assertAlmostEqual(r, 1.7320508075688772)


if __name__ == "__main__":
    unittest.main()
