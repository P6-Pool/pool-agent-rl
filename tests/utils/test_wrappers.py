import unittest
from fastfiz_env.wrappers.utils import cart2sph, sph2deg


class TestFeatures(unittest.TestCase):
    def test_cart2sph(self):
        x, y, z = 1, 1, 1
        r, el, az = cart2sph(x, y, z)
        self.assertAlmostEqual(r, 1.7320508075688772)
        self.assertAlmostEqual(el, 0.9553166181245092)
        self.assertAlmostEqual(az, 0.7853981633974483)

    def test_sph2deg(self):
        r, el, az = 1.7320508075688772, 0.9553166181245092, 0.7853981633974483
        r, theta, phi = sph2deg(r, el, az)
        self.assertAlmostEqual(r, 1.7320508075688772)
        self.assertAlmostEqual(theta, 54.735610317245346)
        self.assertAlmostEqual(phi, 45.00)


if __name__ == "__main__":
    unittest.main()
