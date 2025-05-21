"""Gymnasium wrapper around the Arcade Learning Environment (ALE)."""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any, Literal

import ale_py
import gymnasium
import numpy as np
from ale_py import roms
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict


class AtariEnvStepMetadata(TypedDict):
    """Step info options."""

    lives: int
    episode_frame_number: int
    frame_number: int
    seeds: NotRequired[tuple[int, int]]


class AtariEnv(gymnasium.Env, utils.EzPickle):
    """Gymnasium wrapper around the Arcade Learning Environment (ALE)."""

    # FPS can differ per ROM, therefore, dynamically collect the fps once the game is loaded
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        game: str,
        mode: int | None = None,
        difficulty: int | None = None,
        obs_type: Literal["rgb", "grayscale", "ram"] = "rgb",
        frameskip: tuple[int, int] | int = 4,
        repeat_action_probability: float = 0.25,
        full_action_space: bool = False,
        continuous: bool = False,
        continuous_action_threshold: float = 0.5,
        max_num_frames_per_episode: int | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
        sound_obs: bool = False,
    ):
        """Initialize the ALE for Gymnasium.

        Default parameters are taken from Machado et al., 2018.

        Args:
          game: str => Game to initialize env with, in snake_case.
          mode: Optional[int] => Game mode, see Machado et al., 2018
          difficulty: Optional[int] => Game difficulty,see Machado et al., 2018
          obs_type: str => Observation type in { 'rgb', 'grayscale', 'ram' }
          frameskip: Union[tuple[int, int], int] =>
              Stochastic frameskip as tuple or fixed.
          repeat_action_probability: int =>
              Probability to repeat actions, see Machado et al., 2018
          full_action_space: bool => Use full action space?
          continuous: bool => Use continuous actions?
          continuous_action_threshold: float => threshold used for continuous actions.
          max_num_frames_per_episode: int => Max number of frame per epsiode.
              Once `max_num_frames_per_episode` is reached the episode is
              truncated.
          render_mode: str => One of { 'human', 'rgb_array' }.
              If `human` we'll interactively display the screen and enable
              game sounds. This will lock emulation to the ROMs specified FPS
              If `rgb_array` we'll return the `rgb` key in step metadata with
              the current environment RGB frame.
          sound_obs: bool => Add the sound from the frame to the observation.

        Note:
          - The game must be installed, see ale-import-roms, or ale-py-roms.
          - Frameskip values of (low, high) will enable stochastic frame skip
            which will sample a random frameskip uniformly each action.
          - It is recommended to enable full action space.
            See Machado et al., 2018 for more details.

        References:
            `Revisiting the Arcade Learning Environment: Evaluation Protocols
            and Open Problems for General Agents`, Machado et al., 2018, JAIR
            URL: https://jair.org/index.php/jair/article/view/11182
        """
        if obs_type not in {"rgb", "grayscale", "ram"}:
            raise error.Error(
                f"Invalid observation type: {obs_type}. Expecting: rgb, grayscale, ram."
            )

        if type(frameskip) not in (int, tuple):
            raise error.Error(f"Invalid frameskip type: {type(frameskip)}.")
        if isinstance(frameskip, int) and frameskip <= 0:
            raise error.Error(
                f"Invalid frameskip of {frameskip}, frameskip must be positive."
            )
        elif isinstance(frameskip, tuple) and len(frameskip) != 2:
            raise error.Error(
                f"Invalid stochastic frameskip length of {len(frameskip)}, expected length 2."
            )
        elif isinstance(frameskip, tuple) and frameskip[0] > frameskip[1]:
            raise error.Error(
                "Invalid stochastic frameskip, lower bound is greater than upper bound."
            )
        elif isinstance(frameskip, tuple) and frameskip[0] <= 0:
            raise error.Error(
                "Invalid stochastic frameskip lower bound is greater than upper bound."
            )

        if render_mode is not None and render_mode not in {"rgb_array", "human"}:
            raise error.Error(
                f"Render mode {render_mode} not supported (rgb_array, human)."
            )

        utils.EzPickle.__init__(
            self,
            game=game,
            mode=mode,
            difficulty=difficulty,
            obs_type=obs_type,
            frameskip=frameskip,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            continuous=continuous,
            continuous_action_threshold=continuous_action_threshold,
            max_num_frames_per_episode=max_num_frames_per_episode,
            render_mode=render_mode,
            sound_obs=sound_obs,
        )

        # Initialize ALE
        self.ale = ale_py.ALEInterface()

        self._game = game
        self._game_mode = mode
        self._game_difficulty = difficulty

        self._frameskip = frameskip
        self._obs_type = obs_type
        self.render_mode = render_mode
        self.sound_obs = sound_obs

        # Set logger mode to error only
        self.ale.setLoggerMode(ale_py.LoggerMode.Error)
        # Config sticky action prob.
        self.ale.setFloat("repeat_action_probability", repeat_action_probability)

        if max_num_frames_per_episode is not None:
            self.ale.setInt("max_num_frames_per_episode", max_num_frames_per_episode)

        # If render mode is human we can display screen and sound
        if render_mode == "human":
            self.ale.setBool("display_screen", True)
            self.ale.setBool("sound", True)

        self.ale.setBool("sound_obs", self.sound_obs)

        # seed + load
        self.seed_game()
        self.load_game()

        # get the set of legal actions
        if full_action_space or continuous:
            self._action_set = self.ale.getLegalActionSet()
        else:
            self._action_set = self.ale.getMinimalActionSet()

        # action space
        self.continuous = continuous
        self.continuous_action_threshold = continuous_action_threshold
        if continuous:
            # Actions are radius, theta, and fire, where first two are the parameters of polar coordinates.
            self.action_space = spaces.Box(
                low=np.array([0.0, -np.pi, 0.0]).astype(np.float32),
                high=np.array([1.0, np.pi, 1.0]).astype(np.float32),
                dtype=np.float32,
                shape=(3,),
            )
        else:
            self.action_space = spaces.Discrete(len(self._action_set))

        # initialize observation space
        if self._obs_type == "ram":
            self.observation_space = spaces.Box(
                low=0, high=255, dtype=np.uint8, shape=(self.ale.getRAMSize(),)
            )
        elif self._obs_type == "rgb" or self._obs_type == "grayscale":
            (screen_height, screen_width) = self.ale.getScreenDims()
            image_shape = (
                screen_height,
                screen_width,
            )
            if self._obs_type == "rgb":
                image_shape += (3,)
            self.observation_space = spaces.Box(
                low=0, high=255, dtype=np.uint8, shape=image_shape
            )
        else:
            raise error.Error(f"Unrecognized observation type: {self._obs_type}")

        if self.sound_obs:
            self.observation_space = spaces.Dict(
                image=self.observation_space,
                sound=spaces.Box(low=0, high=255, dtype=np.uint8, shape=(512,)),
            )

    def seed_game(self, seed: int | None = None) -> tuple[int, int]:
        """Seeds the internal and ALE RNG."""
        ss = np.random.SeedSequence(seed)
        np_seed, ale_seed = ss.generate_state(n_words=2)
        self._np_random, seed = seeding.np_random(int(np_seed))
        self.ale.setInt("random_seed", np.int32(ale_seed))
        return np_seed, ale_seed

    def load_game(self) -> None:
        """This function initializes the ROM and sets the corresponding mode and difficulty."""
        self.ale.loadROM(roms.get_rom_path(self._game))

        if self._game_mode is not None:
            self.ale.setMode(self._game_mode)
        if self._game_difficulty is not None:
            self.ale.setDifficulty(self._game_difficulty)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, AtariEnvStepMetadata]:
        """Resets environment and returns initial episode observation."""
        super().reset(seed=seed, options=options)

        # sets the seeds if it's specified for both ALE and frameskip np
        # we only want to do this when commanded to, so we don't reset all previous states, statistics, etc.
        seeded_with = None
        if seed is not None:
            seeded_with = self.seed_game(seed)
            self.load_game()

        self.ale.reset_game()

        obs = self._get_obs()
        info = self._get_info()
        if seeded_with is not None:
            info["seeds"] = seeded_with

        return obs, info

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, AtariEnvStepMetadata]:
        """Perform one agent step, i.e., repeats `action` frameskip # of steps.

        Args:
            action: int | np.ndarray =>
                if `continuous=False` -> action index to execute
                if `continuous=True` -> numpy array of r, theta, fire

        Returns:
            tuple[np.ndarray, float, bool, bool, Dict[str, Any]] =>
                observation, reward, terminal, truncation, metadata

        Note: `metadata` contains the keys "lives".
        """
        # If frameskip is a length 2 tuple then it's stochastic
        # frameskip between [frameskip[0], frameskip[1]] uniformly.
        if isinstance(self._frameskip, int):
            frameskip = self._frameskip
        elif isinstance(self._frameskip, tuple):
            frameskip = self.np_random.integers(*self._frameskip)
        else:
            raise error.Error(f"Invalid frameskip type: {self._frameskip}")

        # action formatting
        if self.continuous:
            # compute the x, y, fire of the joystick
            assert isinstance(action, np.ndarray)
            x, y = action[0] * np.cos(action[1]), action[0] * np.sin(action[1])
            action_idx = self.map_action_idx(
                left_center_right=(
                    -int(x < self.continuous_action_threshold)
                    + int(x > self.continuous_action_threshold)
                ),
                down_center_up=(
                    -int(y < self.continuous_action_threshold)
                    + int(y > self.continuous_action_threshold)
                ),
                fire=(action[-1] > self.continuous_action_threshold),
            )

            strength = action[0]
        else:
            action_idx = self._action_set[action]
            strength = 1.0

        # Frameskip
        reward = 0.0
        for _ in range(frameskip):
            reward += self.ale.act(action_idx, strength)

        is_terminal = self.ale.game_over(with_truncation=False)
        is_truncated = self.ale.game_truncated()

        return self._get_obs(), reward, is_terminal, is_truncated, self._get_info()

    def render(self) -> np.ndarray | None:
        """Renders the ALE with `rgb_array` and `human` options."""
        if self.render_mode == "rgb_array":
            return self.ale.getScreenRGB()
        elif self.render_mode == "human":
            return
        else:
            raise error.Error(
                f"Invalid render mode `{self.render_mode}`. "
                "Supported modes: `human`, `rgb_array`."
            )

    def _get_obs(self) -> np.ndarray | dict[str, np.ndarray]:
        """Retrieves the current observation using `obs_type`."""
        if self._obs_type == "ram":
            image_obs = self.ale.getRAM()
        elif self._obs_type == "rgb":
            image_obs = self.ale.getScreenRGB()
        elif self._obs_type == "grayscale":
            image_obs = self.ale.getScreenGrayscale()
        else:
            raise error.Error(
                f"Unrecognized observation type: {self._obs_type}, expected: 'ram', 'rgb' and 'grayscale'."
            )

        if self.sound_obs:
            return {"image": image_obs, "sound": self.ale.getAudio()}
        return image_obs

    def _get_info(self) -> AtariEnvStepMetadata:
        return {
            "lives": self.ale.lives(),
            "episode_frame_number": self.ale.getEpisodeFrameNumber(),
            "frame_number": self.ale.getFrameNumber(),
        }

    @lru_cache(1)
    def get_keys_to_action(self) -> dict[tuple[str, ...], int | np.ndarray]:
        """Return keymapping -> actions for human play.

        Up, down, left and right are wasd keys with fire being space.
        No op is 'e'

        Returns:
            Dictionary of key values to actions
        """
        UP = "w"
        LEFT = "a"
        RIGHT = "d"
        DOWN = "s"
        FIRE = " "
        NOOP = "e"

        mapping = {
            ale_py.Action.NOOP: (NOOP,),
            ale_py.Action.UP: (UP,),
            ale_py.Action.FIRE: (FIRE,),
            ale_py.Action.DOWN: (DOWN,),
            ale_py.Action.LEFT: (LEFT,),
            ale_py.Action.RIGHT: (RIGHT,),
            ale_py.Action.UPFIRE: (UP, FIRE),
            ale_py.Action.DOWNFIRE: (DOWN, FIRE),
            ale_py.Action.LEFTFIRE: (LEFT, FIRE),
            ale_py.Action.RIGHTFIRE: (RIGHT, FIRE),
            ale_py.Action.UPLEFT: (UP, LEFT),
            ale_py.Action.UPRIGHT: (UP, RIGHT),
            ale_py.Action.DOWNLEFT: (DOWN, LEFT),
            ale_py.Action.DOWNRIGHT: (DOWN, RIGHT),
            ale_py.Action.UPLEFTFIRE: (UP, LEFT, FIRE),
            ale_py.Action.UPRIGHTFIRE: (UP, RIGHT, FIRE),
            ale_py.Action.DOWNLEFTFIRE: (DOWN, LEFT, FIRE),
            ale_py.Action.DOWNRIGHTFIRE: (DOWN, RIGHT, FIRE),
        }

        # Map
        #   (key, key, ...) -> action
        if self.continuous:
            raise AttributeError(
                "`get_keys_to_action` can't be provided for this Atari environment as `continuous=True`."
            )
        else:
            return {mapping[action]: i for i, action in enumerate(self._action_set)}

    @staticmethod
    @lru_cache(18)
    def map_action_idx(
        left_center_right: int, down_center_up: int, fire: bool
    ) -> ale_py.Action:
        """Return an action idx given unit actions for underlying env."""
        # no op and fire
        if left_center_right == 0 and down_center_up == 0 and not fire:
            return ale_py.Action.NOOP
        elif left_center_right == 0 and down_center_up == 0 and fire:
            return ale_py.Action.FIRE

        # cardinal no fire
        elif left_center_right == -1 and down_center_up == 0 and not fire:
            return ale_py.Action.LEFT
        elif left_center_right == 1 and down_center_up == 0 and not fire:
            return ale_py.Action.RIGHT
        elif left_center_right == 0 and down_center_up == -1 and not fire:
            return ale_py.Action.DOWN
        elif left_center_right == 0 and down_center_up == 1 and not fire:
            return ale_py.Action.UP

        # cardinal fire
        if left_center_right == -1 and down_center_up == 0 and fire:
            return ale_py.Action.LEFTFIRE
        elif left_center_right == 1 and down_center_up == 0 and fire:
            return ale_py.Action.RIGHTFIRE
        elif left_center_right == 0 and down_center_up == -1 and fire:
            return ale_py.Action.DOWNFIRE
        elif left_center_right == 0 and down_center_up == 1 and fire:
            return ale_py.Action.UPFIRE

        # diagonal no fire
        elif left_center_right == -1 and down_center_up == -1 and not fire:
            return ale_py.Action.DOWNLEFT
        elif left_center_right == 1 and down_center_up == -1 and not fire:
            return ale_py.Action.DOWNRIGHT
        elif left_center_right == -1 and down_center_up == 1 and not fire:
            return ale_py.Action.UPLEFT
        elif left_center_right == 1 and down_center_up == 1 and not fire:
            return ale_py.Action.UPRIGHT

        # diagonal fire
        elif left_center_right == -1 and down_center_up == -1 and fire:
            return ale_py.Action.DOWNLEFTFIRE
        elif left_center_right == 1 and down_center_up == -1 and fire:
            return ale_py.Action.DOWNRIGHTFIRE
        elif left_center_right == -1 and down_center_up == 1 and fire:
            return ale_py.Action.UPLEFTFIRE
        elif left_center_right == 1 and down_center_up == 1 and fire:
            return ale_py.Action.UPRIGHTFIRE

        # just in case
        else:
            raise LookupError(
                "Unexpected action mapping, expected `left_center_right` and `down_center_up` to be in {-1, 0, 1} and `fire` to only be `True` or `False`. "
                f"Received {left_center_right=}, {down_center_up=} and {fire=}."
            )

    def get_action_meanings(self) -> list[str]:
        """Return the meaning of each action."""
        keys = ale_py.Action.__members__.values()
        values = ale_py.Action.__members__.keys()
        mapping = dict(zip(keys, values))
        return [mapping[action] for action in self._action_set]

    def clone_state(self, include_rng: bool = False) -> ale_py.ALEState:
        """Clone emulator state.

        To reproduce identical states, specify `include_rng` to `True`.

        Args:
            include_rng: If to include the system RNG within the state

        Returns:
            The cloned ALE state
        """
        return self.ale.cloneState(include_rng=include_rng)

    def restore_state(self, state: ale_py.ALEState) -> None:
        """Restore emulator state w/o system state."""
        self.ale.restoreState(state)
