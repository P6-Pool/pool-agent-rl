import fastfiz as ff
import vectormath as vmath


class GameBall:
    RADIUS = 0.028575

    def __init__(self, radius: float, number: int, position: vmath.Vector2, state: int):
        self.number = number
        self.position = position
        self.velocity = vmath.Vector2(0, 0)
        self.state = state
        self.striped = number > 8
        self.is_being_dragged = False

        self.white_color = (255, 255, 255)
        self.black_color = (0, 0, 0)

    def update(
        self,
        time_since_shot_start: float,
        shot: ff.Shot,
        sliding_friction_const: float,
        rolling_friction_const: float,
        gravitational_const: float,
    ):
        relevant_states = self._get_relevant_ball_states_from_shot(shot)

        if not relevant_states:
            return

        cur_state: _BallState = relevant_states[0]

        if cur_state.e_time > time_since_shot_start:
            return

        for state in relevant_states:
            if cur_state.e_time < state.e_time <= time_since_shot_start:
                cur_state = state

        time_since_event_start = time_since_shot_start - cur_state.e_time

        def calc_sliding_displacement(delta_time: float) -> vmath.Vector2:
            rotational_velocity: vmath.Vector3 = GameBall.RADIUS * vmath.Vector3(0, 0, cur_state.ang_vel.z).cross(
                cur_state.ang_vel
            )
            relative_velocity = cur_state.vel + vmath.Vector2(rotational_velocity.x, rotational_velocity.y)
            self.velocity = (
                cur_state.vel - delta_time * gravitational_const * sliding_friction_const * relative_velocity.normalize()
            )
            return (
                cur_state.vel * delta_time
                - 0.5 * sliding_friction_const * gravitational_const * delta_time**2 * relative_velocity.normalize()
            )

        def calc_rolling_displacement(delta_time: float) -> vmath.Vector2:
            self.velocity = (
                cur_state.vel - gravitational_const * rolling_friction_const * delta_time * cur_state.vel.copy().normalize()
            )
            return (
                cur_state.vel * delta_time
                - 0.5 * rolling_friction_const * gravitational_const * delta_time**2 * cur_state.vel.copy().normalize()
            )

        displacement = vmath.Vector2(0, 0)

        if cur_state.state == ff.Ball.ROLLING:
            displacement = calc_rolling_displacement(time_since_event_start)
        if cur_state.state == ff.Ball.SLIDING:
            displacement = calc_sliding_displacement(time_since_event_start)

        self.position = displacement + cur_state.pos
        self.state = cur_state.state

    def force_to_end_of_shot_pos(self, shot: ff.Shot):
        relevant_states = self._get_relevant_ball_states_from_shot(shot)
        if relevant_states:
            self.velocity = vmath.Vector2(0, 0)
            self.position = relevant_states[-1].pos
            self.state = relevant_states[-1].state

    def _get_relevant_ball_states_from_shot(self, shot: ff.Shot):
        relevant_states: list[_BallState] = []

        for event in shot.getEventList():
            if event.getBall1() == self.number:
                new_ball_event = _BallState.from_event_and_ball(event, event.getBall1Data())
                relevant_states.append(new_ball_event)
            elif event.getBall2() == self.number:
                new_ball_event = _BallState.from_event_and_ball(event, event.getBall2Data())
                relevant_states.append(new_ball_event)

        return relevant_states


class _BallState:
    def __init__(
        self,
        e_time: float,
        pos: vmath.Vector2,
        vel: vmath.Vector2,
        ang_vel: vmath.Vector3,
        state: int,
        state_str: str,
        event_course: int,
    ):
        self.e_time = e_time
        self.pos = pos
        self.vel = vel
        self.ang_vel = ang_vel
        self.state = state
        self.state_str = state_str
        self.event_course = event_course

    @classmethod
    def from_event_and_ball(cls, event: ff.Event, ball: ff.Ball):
        e_time = event.getTime()
        pos = ball.getPos()
        vel = ball.getVelocity()
        ang_vel = ball.getSpin()
        state = ball.getState()
        state_str = ball.getStateString()
        event_course = event.getType()
        return cls(
            e_time,
            vmath.Vector2([pos.x, pos.y]),
            vmath.Vector2([vel.x, vel.y]),
            vmath.Vector3([ang_vel.x, ang_vel.y, ang_vel.z]),
            state,
            state_str,
            event_course,
        )

    def __str__(self):
        return (
            f"Time: {self.e_time:.3f},\t "
            f"Pos: ({self.pos.x:.3f}, {self.pos.y:.3f}),\t "
            f"Vel: ({self.vel.x:.3f}, {self.vel.y:.3f}),\t "
            f"State: {self.state_str}"
        )
