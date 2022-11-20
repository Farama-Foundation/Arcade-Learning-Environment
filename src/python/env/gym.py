import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ale_py
import ale_py.roms as roms
import ale_py.roms.utils as rom_utils
import numpy as np

import gym
import gym.logger as logger
from gym import error, spaces, utils

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict


class AtariEnvStepMetadata(TypedDict):
    lives: int
    episode_frame_number: int
    frame_number: int
    seeds: NotRequired[Sequence[int]]


class AtariEnv(gym.Env, utils.EzPickle):
    """
    (A)rcade (L)earning (Gym) (Env)ironment.
    A Gym wrapper around the Arcade Learning Environment (ALE).
    """

    # No render modes
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        game: str = "pong",
        mode: Optional[int] = None,
        difficulty: Optional[int] = None,
        obs_type: str = "rgb",
        frameskip: Union[Tuple[int, int], int] = 4,
        repeat_action_probability: float = 0.25,
        full_action_space: bool = False,
        max_num_frames_per_episode: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the ALE for Gym.
        Default parameters are taken from Machado et al., 2018.

        Args:
          game: str => Game to initialize env with.
          mode: Optional[int] => Game mode, see Machado et al., 2018
          difficulty: Optional[int] => Game difficulty,see Machado et al., 2018
          obs_type: str => Observation type in { 'rgb', 'grayscale', 'ram' }
          frameskip: Union[Tuple[int, int], int] =>
              Stochastic frameskip as tuple or fixed.
          repeat_action_probability: int =>
              Probability to repeat actions, see Machado et al., 2018
          full_action_space: bool => Use full action space?
          max_num_frames_per_episode: int => Max number of frame per epsiode.
              Once `max_num_frames_per_episode` is reached the episode is
              truncated.
          render_mode: str => One of { 'human', 'rgb_array' }.
              If `human` we'll interactively display the screen and enable
              game sounds. This will lock emulation to the ROMs specified FPS
              If `rgb_array` we'll return the `rgb` key in step metadata with
              the current environment RGB frame.

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
        if obs_type == "image":
            logger.warn(
                'obs_type "image" should be replaced with the image type, one of: rgb, grayscale'
            )
            obs_type = "rgb"
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
                f"Invalid stochastic frameskip, lower bound is greater than upper bound."
            )
        elif isinstance(frameskip, tuple) and frameskip[0] <= 0:
            raise error.Error(
                f"Invalid stochastic frameskip lower bound is greater than upper bound."
            )

        if render_mode is not None and render_mode not in {"rgb_array", "human"}:
            raise error.Error(
                f"Render mode {render_mode} not supported (rgb_array, human)."
            )

        utils.EzPickle.__init__(
            self,
            game,
            mode,
            difficulty,
            obs_type,
            frameskip,
            repeat_action_probability,
            full_action_space,
            max_num_frames_per_episode,
            render_mode,
        )

        # Initialize ALE
        self.ale = ale_py.ALEInterface()

        self._game = rom_utils.rom_id_to_name(game)

        self._game_mode = mode
        self._game_difficulty = difficulty

        self._frameskip = frameskip
        self._obs_type = obs_type
        self._render_mode = render_mode

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

        # Seed + Load
        self.seed()

        self._action_set = (
            self.ale.getLegalActionSet()
            if full_action_space
            else self.ale.getMinimalActionSet()
        )
        self._action_space = spaces.Discrete(len(self._action_set))

        # Initialize observation type
        if self._obs_type == "ram":
            self._obs_space = spaces.Box(
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
            self._obs_space = spaces.Box(
                low=0, high=255, dtype=np.uint8, shape=image_shape
            )
        else:
            raise error.Error(f"Unrecognized observation type: {self._obs_type}")

    def seed(self, seed: Optional[int] = None) -> Tuple[int, int]:
        """
        Seeds both the internal numpy rng for stochastic frame skip
        as well as the ALE RNG.

        This function must also initialize the ROM and set the corresponding
        mode and difficulty. `seed` may be called to initialize the environment
        during deserialization by Gym so these side-effects must reside here.

        Args:
            seed: int => Manually set the seed for RNG.
        Returns:
            tuple[int, int] => (np seed, ALE seed)
        """
        ss = np.random.SeedSequence(seed)
        seed1, seed2 = ss.generate_state(n_words=2)

        self.np_random = np.random.default_rng(seed1)
        # ALE only takes signed integers for `setInt`, it'll get converted back
        # to unsigned in StellaEnvironment.
        self.ale.setInt("random_seed", seed2.astype(np.int32))

        if not hasattr(roms, self._game):
            raise error.Error(
                f'We\'re Unable to find the game "{self._game}". Note: Gym no longer distributes ROMs. '
                f"If you own a license to use the necessary ROMs for research purposes you can download them "
                f'via `pip install gym[accept-rom-license]`. Otherwise, you should try importing "{self._game}" '
                f'via the command `ale-import-roms`. If you believe this is a mistake perhaps your copy of "{self._game}" '
                "is unsupported. To check if this is the case try providing the environment variable "
                "`PYTHONWARNINGS=default::ImportWarning:ale_py.roms`. For more information see: "
                "https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management"
            )
        self.ale.loadROM(getattr(roms, self._game))

        if self._game_mode is not None:
            self.ale.setMode(self._game_mode)
        if self._game_difficulty is not None:
            self.ale.setDifficulty(self._game_difficulty)

        return seed1, seed2

    def step(
        self,
        action_ind: int,
    ) -> Tuple[np.ndarray, float, bool, bool, AtariEnvStepMetadata]:
        """
        Perform one agent step, i.e., repeats `action` frameskip # of steps.

        Args:
            action_ind: int => Action index to execute

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]] =>
                observation, reward, terminal, metadata

        Note: `metadata` contains the keys "lives" and "rgb" if
              render_mode == 'rgb_array'.
        """
        # Get action enum, terminal bool, metadata
        action = self._action_set[action_ind]

        # If frameskip is a length 2 tuple then it's stochastic
        # frameskip between [frameskip[0], frameskip[1]] uniformly.
        if isinstance(self._frameskip, int):
            frameskip = self._frameskip
        elif isinstance(self._frameskip, tuple):
            frameskip = self.np_random.integers(*self._frameskip)
        else:
            raise error.Error(f"Invalid frameskip type: {self._frameskip}")

        # Frameskip
        reward = 0.0
        for _ in range(frameskip):
            reward += self.ale.act(action)
        is_terminal = self.ale.game_over(with_truncation=False)
        is_truncated = self.ale.game_truncated()

        return self._get_obs(), reward, is_terminal, is_truncated, self._get_info()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, AtariEnvStepMetadata]:
        """
        Resets environment and returns initial observation.
        """
        del options
        # Gym's new seeding API seeds on reset.
        # This will cause the console to be recreated
        # and loose all previous state, e.g., statistics, etc.
        seeded_with = None
        if seed is not None:
            seeded_with = self.seed(seed)

        self.ale.reset_game()
        obs = self._get_obs()

        info = self._get_info()
        if seeded_with is not None:
            info["seeds"] = seeded_with
        return obs, info

    def render(self) -> Any:
        """
        Render is not supported by ALE. We use a paradigm similar to
        Gym3 which allows you to specify `render_mode` during construction.

        For example,
            gym.make("ale-py:Pong-v0", render_mode="human")
        will display the ALE and maintain the proper interval to match the
        FPS target set by the ROM.
        """
        if self._render_mode == "rgb_array":
            return self.ale.getScreenRGB()
        elif self._render_mode == "human":
            pass
        else:
            raise error.Error(
                f"Invalid render mode `{self._render_mode}`. "
                "Supported modes: `human`, `rgb_array`."
            )

    def _get_obs(self) -> np.ndarray:
        """
        Retreives the current observation.
        This is dependent on `self._obs_type`.
        """
        if self._obs_type == "ram":
            return self.ale.getRAM()
        elif self._obs_type == "rgb":
            return self.ale.getScreenRGB()
        elif self._obs_type == "grayscale":
            return self.ale.getScreenGrayscale()
        else:
            raise error.Error(f"Unrecognized observation type: {self._obs_type}")

    def _get_info(self) -> AtariEnvStepMetadata:
        return {
            "lives": self.ale.lives(),
            "episode_frame_number": self.ale.getEpisodeFrameNumber(),
            "frame_number": self.ale.getFrameNumber(),
        }

    def get_keys_to_action(self) -> Dict[Tuple[int], ale_py.Action]:
        """
        Return keymapping -> actions for human play.
        """
        UP = ord("w")
        LEFT = ord("a")
        RIGHT = ord("d")
        DOWN = ord("s")
        FIRE = ord(" ")

        mapping = {
            ale_py.Action.NOOP: (None,),
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
        #   (key, key, ...) -> action_idx
        # where action_idx is the integer value of the action enum
        #
        actions = self._action_set
        return dict(
            zip(
                map(lambda action: tuple(sorted(mapping[action])), actions),
                range(len(actions)),
            )
        )

    def get_action_meanings(self) -> List[str]:
        """
        Return the meaning of each integer action.
        """
        keys = ale_py.Action.__members__.values()
        values = ale_py.Action.__members__.keys()
        mapping = dict(zip(keys, values))
        return [mapping[action] for action in self._action_set]

    def clone_state(self, include_rng=False) -> ale_py.ALEState:
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        return self.ale.cloneState(include_rng=include_rng)

    def restore_state(self, state: ale_py.ALEState) -> None:
        """Restore emulator state w/o system state."""
        self.ale.restoreState(state)

    def clone_full_state(self) -> ale_py.ALEState:
        """Deprecated method which would clone the emulator and system state."""
        logger.warn(
            "`clone_full_state()` is deprecated and will be removed in a future release of `ale-py`. "
            "Please use `clone_state(include_rng=True)` which is equivalent to `clone_full_state`. "
        )
        return self.ale.cloneSystemState()

    def restore_full_state(self, state: ale_py.ALEState) -> None:
        """Restore emulator state w/ system state including pseudorandomness."""
        logger.warn(
            "restore_full_state() is deprecated and will be removed in a future release of `ale-py`. "
            "Please use `restore_state(state)` which will restore the state regardless of being a full or partial state. "
        )
        self.ale.restoreSystemState(state)

    @property
    def action_space(self) -> spaces.Discrete:
        """
        Return Gym's action space.
        """
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        """
        Return Gym's observation space.
        """
        return self._obs_space

    @property
    def render_mode(self) -> str:
        """
        Attribute render_mode to comply Gym API.
        """
        return self._render_mode
