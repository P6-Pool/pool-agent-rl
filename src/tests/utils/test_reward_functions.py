import unittest
import fastfiz as ff
from fastfiz_env.reward_functions.common import *
from fastfiz_env.reward_functions import CombinedReward
from fastfiz_env.utils.fastfiz import create_table_state
import numpy as np


def weight_fn(num_balls: int, current_step: int, max_steps: int | None) -> float:
    if max_steps is None:
        max_steps = 0
    return num_balls + current_step + max_steps


class TestRewardFunctions(unittest.TestCase):
    possible_shot_action = np.array([0, 0, ff.TableState.MAX_THETA - 0.001, 0, 0], dtype=np.float64)
    impossible_shot_action = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    empty_action = np.array([], dtype=np.float64)

    def test_step_pocketed_reward(self):
        table_state = create_table_state(2)
        table_state_pocketed = create_table_state(2)
        table_state_pocketed.setBall(1, ff.Ball.POCKETED_E, ff.Point(0, 0))
        reward = StepPocketedReward()
        reward.reset(table_state)

        self.assertEqual(reward.get_reward(table_state, table_state, self.empty_action), 0)
        self.assertEqual(reward.get_reward(table_state, table_state_pocketed, self.empty_action), 1)

    def test_game_won_reward(self):
        table_state = create_table_state(3)
        table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0, 0))
        table_state.setBall(2, ff.Ball.STATIONARY, ff.Point(0, 0))

        table_state_pocketed = create_table_state(3)
        table_state_pocketed.setBall(1, ff.Ball.POCKETED_E, ff.Point(0, 0))
        table_state_pocketed.setBall(2, ff.Ball.POCKETED_E, ff.Point(0, 0))

        reward = GameWonReward()
        reward.reset(table_state)
        self.assertEqual(reward.get_reward(table_state, table_state, self.empty_action), 0)
        self.assertEqual(reward.get_reward(table_state, table_state_pocketed, self.empty_action), 1)

    def test_constant_reward(self):
        table_state = create_table_state(2)

        reward = ConstantReward(weight=weight_fn, max_episode_steps=10)
        reward.reset(table_state)

        self.assertEqual(reward.get_reward(table_state, table_state, self.empty_action), 2 + 1 + 10)
        self.assertEqual(reward.get_reward(table_state, table_state, self.empty_action), 2 + 2 + 10)

    def test_cue_ball_pocketed_reward(self):
        table_state = create_table_state(2)
        table_state_pocketed = create_table_state(2)
        table_state.setBall(1, ff.Ball.POCKETED_E, ff.Point(0, 0))
        table_state_pocketed.setBall(0, ff.Ball.POCKETED_E, ff.Point(0, 0))
        reward = CueBallPocketedReward()
        reward.reset(table_state)

        self.assertEqual(reward.get_reward(table_state, table_state, self.empty_action), 0)
        self.assertEqual(reward.get_reward(table_state, table_state_pocketed, self.empty_action), 1)

    def test_cue_ball_not_moved_reward(self):
        table_state = create_table_state(2)
        table_state.setBall(0, ff.Ball.STATIONARY, ff.Point(0, 0))
        table_state_moved = create_table_state(2)
        table_state_moved.setBall(0, ff.Ball.STATIONARY, ff.Point(1, 1))
        reward = CueBallNotMovedReward()
        reward.reset(table_state)

        self.assertEqual(reward.get_reward(table_state, table_state, self.empty_action), 1)
        self.assertEqual(reward.get_reward(table_state, table_state_moved, self.empty_action), 0)

    def test_impossible_shot_reward(self):
        table_state = create_table_state(2)
        reward = ImpossibleShotReward()
        reward.reset(table_state)

        self.assertEqual(reward.get_reward(table_state, table_state, self.possible_shot_action), 0)
        self.assertEqual(reward.get_reward(table_state, table_state, self.impossible_shot_action), 1)
        self.assertEqual(reward.get_reward(table_state, table_state, self.empty_action), 1)

    def test_delta_best_total_distance_reward(self):
        # Pocket: [0.,  1.118]
        prev_table_state = create_table_state(2)
        prev_table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.5, 1.118))
        reward = DeltaBestTotalDistanceReward()
        reward.reset(prev_table_state)

        table_state = create_table_state(2)
        table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.25, 1.118))

        self.assertEqual(reward.get_reward(prev_table_state, prev_table_state, self.empty_action), 0)
        self.assertEqual(
            reward.get_reward(prev_table_state, table_state, self.empty_action),
            0.25,
        )
        self.assertEqual(
            reward.get_reward(prev_table_state, table_state, self.empty_action),
            0,
        )  # Should be 0 because of the min_total_dist update

    def test_total_distance_reward(self):
        # Pocket: [0.,  1.118]
        prev_table_state = create_table_state(2)
        prev_table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.5, 1.118))
        reward = TotalDistanceReward()
        reward.reset(prev_table_state)

        table_state = create_table_state(2)
        table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.25, 1.118))

        self.assertEqual(
            reward.get_reward(prev_table_state, prev_table_state, self.empty_action),
            0.5,
        )
        self.assertEqual(
            reward.get_reward(prev_table_state, table_state, self.empty_action),
            0.25,
        )

    def test_combined_reward(self):
        rewards_functions = [
            StepPocketedReward(1),
            DeltaBestTotalDistanceReward(4),
            GameWonReward(1),
            ImpossibleShotReward(1),
            CueBallNotMovedReward(1),
            CueBallPocketedReward(1),
            ConstantReward(1.3),
        ]
        # Expected out: 0, 0.25 * 4, 0, 0, 1, 0, 1*1.3 = 3.3

        reward_function = CombinedReward(reward_functions=rewards_functions)
        prev_table_state = create_table_state(2)
        prev_table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.5, 1.118))
        reward_function.reset(prev_table_state)

        table_state = create_table_state(2)
        table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.25, 1.118))

        self.assertEqual(
            reward_function.get_reward(prev_table_state, table_state, self.empty_action),
            3.3,
        )

    def test_binary_reward_no_short_circuit(self):
        rewards_functions = [
            ImpossibleShotReward(10, short_circuit=False),
            ConstantReward(5),
        ]
        reward_function = CombinedReward(reward_functions=rewards_functions, short_circuit=True)
        table_state = create_table_state(2)
        reward_function.reset(table_state)

        self.assertEqual(
            reward_function.get_reward(table_state, table_state, self.possible_shot_action),
            5,
        )
        self.assertEqual(
            reward_function.get_reward(table_state, table_state, self.impossible_shot_action),
            15,
        )

    def test_binary_reward_short_circuit(self):
        rewards_functions = [
            ImpossibleShotReward(10, short_circuit=True),
            ConstantReward(5),
        ]
        reward_function = CombinedReward(reward_functions=rewards_functions, short_circuit=True)
        table_state = create_table_state(2)
        reward_function.reset(table_state)

        self.assertEqual(
            reward_function.get_reward(table_state, table_state, self.possible_shot_action),
            5,
        )
        self.assertEqual(
            reward_function.get_reward(table_state, table_state, self.impossible_shot_action),
            10,
        )

    def test_weights(self):
        table_state = create_table_state(3)
        reward = ConstantReward(weight=NegativeConstantWeightMaxSteps, max_episode_steps=10)
        reward.reset(table_state)

        self.assertEqual(
            reward.get_reward(table_state, table_state, self.empty_action),
            -1 / 10,
        )

        reward = ConstantReward(weight=ConstantWeightCurrentStep, max_episode_steps=10)
        reward.reset(table_state)

        self.assertEqual(
            reward.get_reward(table_state, table_state, self.empty_action),
            1,
        )

        reward = ConstantReward(weight=ConstantWeightBalls, max_episode_steps=10)
        reward.reset(table_state)

        self.assertEqual(
            reward.get_reward(table_state, table_state, self.empty_action),
            1 / 2,
        )
        reward = ConstantReward(weight=ConstantWeightNumBalls, max_episode_steps=10)
        reward.reset(table_state)

        self.assertEqual(
            reward.get_reward(table_state, table_state, self.empty_action),
            1 / 3,
        )
        reward_fn = ConstantReward(weight=ConstantWeightMaxSteps)
        reward = CombinedReward(reward_functions=[reward_fn], max_episode_steps=5)
        reward.reset(table_state)

        self.assertEqual(
            reward.get_reward(table_state, table_state, self.empty_action),
            1 / 5,
        )

        reward = ConstantReward(weight=ConstantWeightCurrentStep, max_episode_steps=10)
        reward.reset(table_state)
        for _ in range(1, 3):
            reward.get_reward(table_state, table_state, self.empty_action)

        self.assertEqual(
            reward.get_reward(table_state, table_state, self.empty_action),
            1 / 3,
        )


if __name__ == "__main__":
    unittest.main()
