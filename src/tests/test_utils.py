import unittest
import fastfiz as ff
from fastfiz_env.utils.fastfiz import (
    create_table_state,
    num_balls_in_play,
    any_ball_has_moved,
    get_ball_positions,
)


class TestUtilsFastFiz(unittest.TestCase):
    def test_create_table_state(self):
        num_balls = 15
        table_state = create_table_state(num_balls)

        self.assertEqual(num_balls_in_play(table_state), num_balls + 1)

    def test_any_ball_has_moved(self):
        num_balls = 15
        table_state = create_table_state(num_balls)
        table_state.setBall(0, ff.Ball.STATIONARY, ff.Point(1.0, 1.0))
        prev_ball_positions = get_ball_positions(table_state)
        table_state.setBall(0, ff.Ball.STATIONARY, ff.Point(0.0, 0.0))
        ball_positions = get_ball_positions(table_state)
        self.assertTrue(any_ball_has_moved(prev_ball_positions, ball_positions))


if __name__ == "__main__":
    unittest.main()