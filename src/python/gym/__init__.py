# from ale_py.gym.utils import register_legacy_gym_envs
from ale_py.gym.environment import ALGymEnv

# We don't export anything
__all__ = ["ALGymEnv"]

# Once a proper plugin system is implemented in Gym,
# e.g., https://github.com/openai/gym/issues/2345
# We'll be able to register the legacy environments ourself.
# register_legacy_gym_envs()
