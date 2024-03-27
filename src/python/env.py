from __future__ import annotations

import sys
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
    lives: int
    episode_frame_number: int
    frame_number: int
    seeds: NotRequired[tuple[int, int]]


class AtariEnv(gymnasium.Env, utils.EzPickle):
    """
    (A)rcade (L)earning (Gym) (Env)ironment.
    A Gym wrapper around the Arcade Learning Environment (ALE).
    """

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
        max_num_frames_per_episode: int | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
    ):
        """
        Initialize the ALE for Gymnasium.
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
            max_num_frames_per_episode=max_num_frames_per_episode,
            render_mode=render_mode,
        )

        # Initialize ALE
        self.ale = ale_py.ALEInterface()

        self._game = game
        self._game_mode = mode
        self._game_difficulty = difficulty

        self._frameskip = frameskip
        self._obs_type = obs_type
        self.render_mode = render_mode

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

        # seed + load
        self.seed_game()
        self.load_game()

        # initialize action space
        self._action_set = (
            self.ale.getLegalActionSet()
            if full_action_space
            else self.ale.getMinimalActionSet()
        )
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

    def reset(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, AtariEnvStepMetadata]:
        """Resets environment and returns initial observation."""
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
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, AtariEnvStepMetadata]:
        """
        Perform one agent step, i.e., repeats `action` frameskip # of steps.

        Args:
            action_ind: int => Action index to execute

        Returns:
            tuple[np.ndarray, float, bool, bool, Dict[str, Any]] =>
                observation, reward, terminal, truncation, metadata

        Note: `metadata` contains the keys "lives" and "rgb" if
              render_mode == 'rgb_array'.
        """
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
            reward += self.ale.act(self._action_set[action])
        is_terminal = self.ale.game_over(with_truncation=False)
        is_truncated = self.ale.game_truncated()

        return self._get_obs(), reward, is_terminal, is_truncated, self._get_info()

    def render(self) -> np.ndarray | None:
        """
        Render is not supported by ALE. We use a paradigm similar to
        Gym3 which allows you to specify `render_mode` during construction.

        For example,
            gym.make("ale-py:Pong-v0", render_mode="human")
        will display the ALE and maintain the proper interval to match the
        FPS target set by the ROM.
        """
        if self.render_mode == "rgb_array":
            return self.ale.getScreenRGB()
        elif self.render_mode == "human":
            return
        else:
            raise error.Error(
                f"Invalid render mode `{self.render_mode}`. "
                "Supported modes: `human`, `rgb_array`."
            )

    def _get_obs(self) -> np.ndarray:
        """
        Retrieves the current observation.
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

    def get_keys_to_action(self) -> dict[tuple[int, ...], ale_py.Action]:
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
        return dict(
            zip(
                map(lambda action: tuple(sorted(mapping[action])), self._action_set),
                range(len(self._action_set)),
            )
        )

    def get_action_meanings(self) -> list[str]:
        """
        Return the meaning of each integer action.
        """
        keys = ale_py.Action.__members__.values()
        values = ale_py.Action.__members__.keys()
        mapping = dict(zip(keys, values))
        return [mapping[action] for action in self._action_set]

    def clone_state(self, include_rng: bool = False) -> ale_py.ALEState:
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        return self.ale.cloneState(include_rng=include_rng)

    def restore_state(self, state: ale_py.ALEState) -> None:
        """Restore emulator state w/o system state."""
        self.ale.restoreState(state)
