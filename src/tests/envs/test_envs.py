import unittest
from fastfiz_env.envs import BaseRLFastFiz
from fastfiz_env.utils.reward_functions.common import ConstantReward


class TestBaseRLFastFiz(unittest.TestCase):
    def test_init(self):
        num_balls = 16
        env = BaseRLFastFiz(num_balls=num_balls)
        self.assertEqual(env.observation_space.shape, (16, 2))
        self.assertEqual(env.action_space.shape, (5,))

    def test_reset(self):
        num_balls = 16
        env = BaseRLFastFiz(num_balls=num_balls)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (16, 2))
        self.assertEqual(info, {"is_success": False})

    def test_step(self):
        num_balls = 16
        env = BaseRLFastFiz(num_balls=num_balls, reward_function=ConstantReward())
        env.reset()
        action = [0, 0, 60, 0, 0]
        obs, reward, done, truncated, info = env.step(action)
        self.assertEqual(obs.shape, (16, 2))
        self.assertEqual(reward, 1)
        self.assertEqual(done, False)
        self.assertEqual(truncated, False)
        self.assertEqual(info, {"is_success": False})


if __name__ == "__main__":
    unittest.main()
