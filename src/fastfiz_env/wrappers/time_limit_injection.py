import gymnasium as gym


def get_wrapper_attr(env, wrapper, attr):
    if hasattr(env, "env"):
        if isinstance(env, wrapper):
            return getattr(env, attr)
        get_wrapper_attr(env.env, wrapper, attr)
    return None


def inject_attribute_into_base_env(env, attribute_name, attribute_value):
    if hasattr(env, "env"):
        return inject_attribute_into_base_env(env.env, attribute_name, attribute_value)
    setattr(env, attribute_name, attribute_value)


class TimeLimitInjectionWrapper(gym.Wrapper):
    """
    A wrapper that injects the max_episode_steps and elapsed_steps attributes from TimeLimit wrapper into the base environment.
    """

    def __init__(self, env):
        """Wrap a gymnasium environment.
        Args:
            env (gym.Env): The gymnasium environment.
        """
        super().__init__(env)

        # Retrieve max_episode_steps from potentially nested TimeLimit wrappers.
        max_episode_steps = get_wrapper_attr(self.env, gym.wrappers.TimeLimit, "_max_episode_steps")  # type: ignore
        elapsed_steps = get_wrapper_attr(self.env, gym.wrappers.TimeLimit, "_elapsed_steps")  # type: ignore

        # Inject the max_episode_steps attribute into the base environment.
        inject_attribute_into_base_env(
            self.env, "_max_episode_steps", max_episode_steps
        )
        inject_attribute_into_base_env(self.env, "_elapsed_steps", elapsed_steps)
