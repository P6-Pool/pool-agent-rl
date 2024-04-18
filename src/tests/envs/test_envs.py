import unittest
from fastfiz_env.envs import SimpleFastFiz
from fastfiz_env.reward_functions.common import ConstantReward
from fastfiz_env.wrappers import TimeLimitInjectionWrapper


class TestSimpleFastFiz(unittest.TestCase):
    def test_init(self):
        num_balls = 16
        env = SimpleFastFiz(num_balls=num_balls)
        env = TimeLimitInjectionWrapper(env)
        self.assertEqual(env.observation_space.shape, (16, 2))
        self.assertEqual(env.action_space.shape, (3,))

    def test_reset(self):
        num_balls = 16
        env = SimpleFastFiz(num_balls=num_balls)
        env = TimeLimitInjectionWrapper(env)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (16, 2))
        self.assertEqual(info, {"is_success": False})

    def test_step(self):
        num_balls = 16
        env = SimpleFastFiz(num_balls=num_balls, reward_function=ConstantReward())
        env = TimeLimitInjectionWrapper(env)
        env.reset()
        action = [0, 0, 60, 0, 0]
        obs, reward, done, truncated, info = env.step(action)
        self.assertEqual(obs.shape, (16, 2))
        self.assertEqual(reward, 1)
        self.assertEqual(done, True)  # Will terminate as no balls were pocketed
        self.assertEqual(truncated, False)
        self.assertEqual(info, {"is_success": False})


if __name__ == "__main__":
    unittest.main()
