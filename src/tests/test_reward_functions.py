import unittest
from fastfiz_env.utils.reward_functions.common import *
from fastfiz_env.utils.reward_functions import CombinedReward
from fastfiz_env.utils.fastfiz import (
    create_table_state,
    get_ball_positions,
    pocket_centers,
)
import fastfiz as ff


class TestRewardFunctions(unittest.TestCase):
    def test_step_pocketed_reward(self):
        reward = StepPocketedReward()
        table_state = create_table_state(2)
        table_state_pocketed = create_table_state(2)
        table_state_pocketed.setBall(1, ff.Ball.POCKETED_E, ff.Point(0, 0))

        self.assertEqual(reward.get_reward(table_state, table_state, False), 0)
        self.assertEqual(reward.get_reward(table_state, table_state_pocketed, True), 1)

    def test_game_won_reward(self):
        table_state = create_table_state(3)
        reward = GameWonReward()
        reward.reset(table_state)
        table_state_pocketed = create_table_state(2)
        table_state_pocketed.setBall(1, ff.Ball.POCKETED_E, ff.Point(0, 0))
        table_state_pocketed.setBall(2, ff.Ball.POCKETED_E, ff.Point(0, 0))

        self.assertEqual(reward.get_reward(table_state, table_state, False), 0)
        self.assertEqual(reward.get_reward(table_state, table_state_pocketed, True), 1)

    def test_constant_reward(self):
        reward = ConstantReward()
        table_state = create_table_state(2)
        self.assertEqual(reward.get_reward(table_state, table_state, False), 1)
        self.assertEqual(reward.get_reward(table_state, table_state, True), 1)

    def test_cue_ball_pocketed_reward(self):
        reward = CueBallPocketedReward()
        table_state = create_table_state(2)
        table_state_pocketed = create_table_state(2)
        table_state.setBall(1, ff.Ball.POCKETED_E, ff.Point(0, 0))
        table_state_pocketed.setBall(0, ff.Ball.POCKETED_E, ff.Point(0, 0))

        self.assertEqual(reward.get_reward(table_state, table_state, False), 0)
        self.assertEqual(reward.get_reward(table_state, table_state_pocketed, True), 1)

    def test_cue_ball_not_moved_reward(self):
        reward = CueBallNotMovedReward()
        table_state = create_table_state(2)
        table_state.setBall(0, ff.Ball.STATIONARY, ff.Point(0, 0))
        table_state_moved = create_table_state(2)
        table_state_moved.setBall(0, ff.Ball.STATIONARY, ff.Point(1, 1))

        self.assertEqual(reward.get_reward(table_state, table_state, False), 1)
        self.assertEqual(reward.get_reward(table_state, table_state_moved, True), 0)

    def test_impossible_shot_reward(self):
        table_state = create_table_state(2)
        reward = ImpossibleShotReward()

        self.assertEqual(reward.get_reward(table_state, table_state, False), 0)
        self.assertEqual(reward.get_reward(table_state, table_state, True), 1)

    def test_best_total_distance_reward(self):
        # Pocket: [0.,  1.118]
        prev_table_state = create_table_state(2)
        prev_table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.5, 1.118))
        reward = BestDeltaDistanceReward()
        reward.reset(prev_table_state)

        table_state = create_table_state(2)
        table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.25, 1.118))

        self.assertEqual(
            reward.get_reward(prev_table_state, prev_table_state, False), 0
        )
        self.assertEqual(
            reward.get_reward(prev_table_state, table_state, False),
            0.25,
        )
        self.assertEqual(
            reward.get_reward(prev_table_state, table_state, False),
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
            reward.get_reward(prev_table_state, prev_table_state, False), 0.5
        )
        self.assertEqual(
            reward.get_reward(prev_table_state, table_state, False),
            0.25,
        )

    def test_combined_reward(self):
        rewards_functions = [
            StepPocketedReward(),
            BestDeltaDistanceReward(),
            GameWonReward(),
            ImpossibleShotReward(),
            CueBallNotMovedReward(),
            CueBallPocketedReward(),
            ConstantReward(),
        ]
        reward_weights = [1, 4, 1, 1, 1, 1, 1.3]
        # Expected out: 0, 0.25 * 4, 0, 0, 1, 0, 1*1.3 = 3.3

        reward_function = CombinedReward(rewards_functions, reward_weights)
        prev_table_state = create_table_state(2)
        prev_table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.5, 1.118))
        reward_function.reset(prev_table_state)

        table_state = create_table_state(2)
        table_state.setBall(1, ff.Ball.STATIONARY, ff.Point(0.25, 1.118))

        self.assertEqual(
            reward_function.get_reward(prev_table_state, table_state, True), 3.3
        )

    def test_binary_reward_no_short_circuit(self):
        rewards_functions = [
            ImpossibleShotReward(short_circuit=False),
            ConstantReward(),
        ]
        reward_weights = [10, 5]
        reward_function = CombinedReward(
            rewards_functions, reward_weights, short_circuit=True
        )
        table_state = create_table_state(2)
        reward_function.reset(table_state)

        self.assertEqual(reward_function.get_reward(table_state, table_state, False), 5)
        self.assertEqual(reward_function.get_reward(table_state, table_state, True), 15)

    def test_binary_reward_short_circuit(self):
        rewards_functions = [
            ImpossibleShotReward(short_circuit=True),
            ConstantReward(),
        ]
        reward_weights = [10, 5]
        reward_function = CombinedReward(
            rewards_functions, reward_weights, short_circuit=True
        )
        table_state = create_table_state(2)
        reward_function.reset(table_state)

        self.assertEqual(reward_function.get_reward(table_state, table_state, False), 5)
        self.assertEqual(reward_function.get_reward(table_state, table_state, True), 10)


if __name__ == "__main__":
    unittest.main()
