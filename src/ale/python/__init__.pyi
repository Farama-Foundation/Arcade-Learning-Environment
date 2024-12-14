import os
from typing import List, Optional, overload

import numpy as np
import numpy.typing as npt
from ale_py import _ale_py

__all__ = [
    "Action",
    "ALEInterface",
    "ALEState",
    "LoggerMode",
    "SDL_SUPPORT",
]

class LoggerMode:
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """

    @property
    def value(self) -> int:
        """
        :type: int
        """
    Error: _ale_py.LoggerMode  # value = <LoggerMode.Error: 2>
    Info: _ale_py.LoggerMode  # value = <LoggerMode.Info: 0>
    Warning: _ale_py.LoggerMode  # value = <LoggerMode.Warning: 1>
    __members__: dict  # value = {'Info': <LoggerMode.Info: 0>, 'Warning': <LoggerMode.Warning: 1>, 'Error': <LoggerMode.Error: 2>}
    pass

class Action:
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """

    @property
    def value(self) -> int:
        """
        :type: int
        """
    PLAYER_A_DOWN: _ale_py.Action  # value = <Action.PLAYER_A_DOWN: 5>
    PLAYER_A_DOWNFIRE: _ale_py.Action  # value = <Action.PLAYER_A_DOWNFIRE:  >
    PLAYER_A_DOWNLEFT: _ale_py.Action  # value = <Action.PLAYER_A_DOWNLEFT: 9>
    PLAYER_A_DOWNLEFTFIRE: _ale_py.Action  # value = <Action.PLAYER_A_DOWNLEFTFIRE: 17>
    PLAYER_A_DOWNRIGHT: _ale_py.Action  # value = <Action.PLAYER_A_DOWNRIGHT: 8>
    PLAYER_A_DOWNRIGHTFIRE: _ale_py.Action  # value = <Action.PLAYER_A_DOWNRIGHTFIRE: 16>
    PLAYER_A_FIRE: _ale_py.Action  # value = <Action.PLAYER_A_FIRE: 1>
    PLAYER_A_LEFT: _ale_py.Action  # value = <Action.PLAYER_A_LEFT: 4>
    PLAYER_A_LEFTFIRE: _ale_py.Action  # value = <Action.PLAYER_A_LEFTFIRE: 12>
    PLAYER_A_NOOP: _ale_py.Action  # value = <Action.PLAYER_A_NOOP: 0>
    PLAYER_A_RIGHT: _ale_py.Action  # value = <Action.PLAYER_A_RIGHT: 3>
    PLAYER_A_RIGHTFIRE: _ale_py.Action  # value = <Action.PLAYER_A_RIGHTFIRE: 11>
    PLAYER_A_UP: _ale_py.Action  # value = <Action.PLAYER_A_UP: 2>
    PLAYER_A_UPFIRE: _ale_py.Action  # value = <Action.PLAYER_A_UPFIRE: 10>
    PLAYER_A_UPLEFT: _ale_py.Action  # value = <Action.PLAYER_A_UPLEFT: 7>
    PLAYER_A_UPLEFTFIRE: _ale_py.Action  # value = <Action.PLAYER_A_UPLEFTFIRE: 15>
    PLAYER_A_UPRIGHT: _ale_py.Action  # value = <Action.PLAYER_A_UPRIGHT: 6>
    PLAYER_A_UPRIGHTFIRE: _ale_py.Action  # value = <Action.PLAYER_A_UPRIGHTFIRE: 14>
    PLAYER_B_DOWN: _ale_py.Action  # value = <Action.PLAYER_B_DOWN: 23>
    PLAYER_B_DOWNFIRE: _ale_py.Action  # value = <Action.PLAYER_B_DOWNFIRE: 31>
    PLAYER_B_DOWNLEFT: _ale_py.Action  # value = <Action.PLAYER_B_DOWNLEFT: 27>
    PLAYER_B_DOWNLEFTFIRE: _ale_py.Action  # value = <Action.PLAYER_B_DOWNLEFTFIRE: 35>
    PLAYER_B_DOWNRIGHT: _ale_py.Action  # value = <Action.PLAYER_B_DOWNRIGHT: 26>
    PLAYER_B_DOWNRIGHTFIRE: _ale_py.Action  # value = <Action.PLAYER_B_DOWNRIGHTFIRE: 34>
    PLAYER_B_FIRE: _ale_py.Action  # value = <Action.PLAYER_B_FIRE: 19>
    PLAYER_B_LEFT: _ale_py.Action  # value = <Action.PLAYER_B_LEFT: 22>
    PLAYER_B_LEFTFIRE: _ale_py.Action  # value = <Action.PLAYER_B_LEFTFIRE: 30>
    PLAYER_B_NOOP: _ale_py.Action  # value = <Action.PLAYER_B_NOOP: 18>
    PLAYER_B_RIGHT: _ale_py.Action  # value = <Action.PLAYER_B_RIGHT: 21>
    PLAYER_B_RIGHTFIRE: _ale_py.Action  # value = <Action.PLAYER_B_RIGHTFIRE: 29>
    PLAYER_B_UP: _ale_py.Action  # value = <Action.PLAYER_B_UP: 20>
    PLAYER_B_UPFIRE: _ale_py.Action  # value = <Action.PLAYER_B_UPFIRE: 28>
    PLAYER_B_UPLEFT: _ale_py.Action  # value = <Action.PLAYER_B_UPLEFT: 25>
    PLAYER_B_UPLEFTFIRE: _ale_py.Action  # value = <Action.PLAYER_B_UPLEFTFIRE: 33>
    PLAYER_B_UPRIGHT: _ale_py.Action  # value = <Action.PLAYER_B_UPRIGHT: 24>
    PLAYER_B_UPRIGHTFIRE: _ale_py.Action  # value = <Action.PLAYER_B_UPRIGHTFIRE: 32>
    __members__: dict  # value = {'PLAYER_A_NOOP': <Action.NOOP: 0>, 'PLAYER_A_FIRE': <Action.FIRE: 1>, 'PLAYER_A_UP': <Action.UP: 2>, 'PLAYER_A_RIGHT': <Action.RIGHT: 3>, 'PLAYER_A_LEFT': <Action.LEFT: 4>, 'PLAYER_A_DOWN': <Action.DOWN: 5>, 'PLAYER_A_UPRIGHT': <Action.UPRIGHT: 6>, 'PLAYER_A_UPLEFT': <Action.UPLEFT: 7>, 'PLAYER_A_DOWNRIGHT': <Action.DOWNRIGHT: 8>, 'PLAYER_A_DOWNLEFT': <Action.DOWNLEFT: 9>, 'PLAYER_A_UPFIRE': <Action.UPFIRE: 10>, 'PLAYER_A_RIGHTFIRE': <Action.RIGHTFIRE: 11>, 'PLAYER_A_LEFTFIRE': <Action.LEFTFIRE: 12>, 'PLAYER_A_DOWNFIRE': <Action.DOWNFIRE: 13>, 'PLAYER_A_UPRIGHTFIRE': <Action.UPRIGHTFIRE: 14>, 'PLAYER_A_UPLEFTFIRE': <Action.UPLEFTFIRE: 15>, 'PLAYER_A_DOWNRIGHTFIRE': <Action.DOWNRIGHTFIRE: 16>, 'PLAYER_A_DOWNLEFTFIRE': <Action.DOWNLEFTFIRE: 17> 'PLAYER_B_NOOP': <Action.NOOP: 18>, 'PLAYER_B_FIRE': <Action.FIRE: 19>, 'PLAYER_B_UP': <Action.UP: 20>, 'PLAYER_B_RIGHT': <Action.RIGHT: 21>, 'PLAYER_B_LEFT': <Action.LEFT: 22>, 'PLAYER_B_DOWN': <Action.DOWN: 23>, 'PLAYER_B_UPRIGHT': <Action.UPRIGHT: 24>, 'PLAYER_B_UPLEFT': <Action.UPLEFT: 25>, 'PLAYER_B_DOWNRIGHT': <Action.DOWNRIGHT: 26>, 'PLAYER_B_DOWNLEFT': <Action.DOWNLEFT: 27>, 'PLAYER_B_UPFIRE': <Action.UPFIRE: 28>, 'PLAYER_B_RIGHTFIRE': <Action.RIGHTFIRE: 29>, 'PLAYER_B_LEFTFIRE': <Action.LEFTFIRE: 30>, 'PLAYER_B_DOWNFIRE': <Action.DOWNFIRE: 31>, 'PLAYER_B_UPRIGHTFIRE': <Action.UPRIGHTFIRE: 32>, 'PLAYER_B_UPLEFTFIRE': <Action.UPLEFTFIRE: 33>, 'PLAYER_B_DOWNRIGHTFIRE': <Action.DOWNRIGHTFIRE: 34>, 'PLAYER_B_DOWNLEFTFIRE': <Action.DOWNLEFTFIRE: 35>}
    pass

class ALEState:
    def __eq__(self, other: ALEState) -> bool: ...
    def __getstate__(self) -> tuple: ...
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, state: ALEState, serialized: str) -> None: ...
    @overload
    def __init__(self, serialized: str) -> None: ...
    def __setstate__(self, state: tuple) -> None: ...
    def equals(self, other: ALEState) -> bool: ...
    def getCurrentMode(self) -> int: ...
    def getDifficulty(self) -> int: ...
    def getEpisodeFrameNumber(self) -> int: ...
    def getFrameNumber(self) -> int: ...
    def serialize(self) -> str: ...
    __hash__ = None  # type: ignore
    pass

class ALEInterface:
    def __init__(self) -> None: ...
    @overload
    def act(self, a_action: Action, a_paddle_strength: float = 1.0, b_action: Action = Action.PLAYER_B_NOOP, b_paddle_strength: float = 1.0) -> int: ...
    @overload
    def act(self, a_action: int, a_paddle_strength: float = 1.0, b_action: int = 18, b_paddle_strength: float = 1.0) -> int: ...
    def cloneState(self, *, include_rng: bool = False) -> ALEState: ...
    def cloneSystemState(self) -> ALEState: ...
    def game_over(self, *, with_truncation: bool = True) -> bool: ...
    def game_truncated(self) -> bool: ...
    def getAvailableDifficulties(self) -> List[int]: ...
    def getAvailableModes(self) -> List[int]: ...
    def getBool(self, key: str) -> bool: ...
    def getEpisodeFrameNumber(self) -> int: ...
    def getFloat(self, key: str) -> float: ...
    def getFrameNumber(self) -> int: ...
    def getInt(self, key: str) -> int: ...
    def getLegalActionSet(self) -> List[Action]: ...
    def getMinimalActionSet(self) -> List[Action]: ...
    @overload
    def getRAM(self) -> npt.NDArray[np.uint8]: ...
    @overload
    def getRAM(self, arg0: npt.NDArray[np.uint8]) -> None: ...
    def getRAMSize(self) -> int: ...
    @overload
    def getAudio(self) -> npt.NDArray[np.uint8]: ...
    @overload
    def getAudio(self, array: npt.NDArray[np.uint8]) -> None: ...
    def getAudioSize(self) -> int: ...
    @overload
    def getScreen(self) -> npt.NDArray[np.uint8]: ...
    @overload
    def getScreen(self, array: npt.NDArray[np.uint8]) -> None: ...
    def getScreenDims(self) -> tuple: ...
    @overload
    def getScreenGrayscale(self) -> npt.NDArray[np.uint8]: ...
    @overload
    def getScreenGrayscale(self, array: npt.NDArray[np.uint8]) -> None: ...
    @overload
    def getScreenRGB(self) -> npt.NDArray[np.uint8]: ...
    @overload
    def getScreenRGB(self, array: npt.NDArray[np.uint8]) -> None: ...
    def getString(self, key: str) -> str: ...
    @staticmethod
    @overload
    def isSupportedROM(rom: os.PathLike) -> Optional[str]: ...
    @staticmethod
    @overload
    def isSupportedROM(rom: str) -> Optional[str]: ...
    def lives(self) -> int: ...
    @overload
    def loadROM(self, rom: os.PathLike) -> None: ...
    @overload
    def loadROM(self, rom: str) -> None: ...
    def reset_game(self) -> None: ...
    def restoreState(self, state: ALEState) -> None: ...
    def restoreSystemState(self, state: ALEState) -> None: ...
    def saveScreenPNG(self, path: str) -> None: ...
    def setBool(self, key: str, value: bool) -> None: ...
    def setDifficulty(self, difficulty: int) -> None: ...
    def setFloat(self, key: str, value: float) -> None: ...
    def setInt(self, key: str, value: int) -> None: ...
    @staticmethod
    def setLoggerMode(mode: LoggerMode) -> None: ...
    def setMode(self, mode: int) -> None: ...
    def setRAM(self, index: int, value: int) -> None: ...
    def setString(self, key: str, value: str) -> None: ...
    pass

SDL_SUPPORT: bool
__version__: str
