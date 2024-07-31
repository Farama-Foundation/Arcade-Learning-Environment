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
    DOWN: _ale_py.Action  # value = <Action.DOWN: 5>
    DOWNFIRE: _ale_py.Action  # value = <Action.DOWNFIRE: 13>
    DOWNLEFT: _ale_py.Action  # value = <Action.DOWNLEFT: 9>
    DOWNLEFTFIRE: _ale_py.Action  # value = <Action.DOWNLEFTFIRE: 17>
    DOWNRIGHT: _ale_py.Action  # value = <Action.DOWNRIGHT: 8>
    DOWNRIGHTFIRE: _ale_py.Action  # value = <Action.DOWNRIGHTFIRE: 16>
    FIRE: _ale_py.Action  # value = <Action.FIRE: 1>
    LEFT: _ale_py.Action  # value = <Action.LEFT: 4>
    LEFTFIRE: _ale_py.Action  # value = <Action.LEFTFIRE: 12>
    NOOP: _ale_py.Action  # value = <Action.NOOP: 0>
    RIGHT: _ale_py.Action  # value = <Action.RIGHT: 3>
    RIGHTFIRE: _ale_py.Action  # value = <Action.RIGHTFIRE: 11>
    UP: _ale_py.Action  # value = <Action.UP: 2>
    UPFIRE: _ale_py.Action  # value = <Action.UPFIRE: 10>
    UPLEFT: _ale_py.Action  # value = <Action.UPLEFT: 7>
    UPLEFTFIRE: _ale_py.Action  # value = <Action.UPLEFTFIRE: 15>
    UPRIGHT: _ale_py.Action  # value = <Action.UPRIGHT: 6>
    UPRIGHTFIRE: _ale_py.Action  # value = <Action.UPRIGHTFIRE: 14>
    __members__: dict  # value = {'NOOP': <Action.NOOP: 0>, 'FIRE': <Action.FIRE: 1>, 'UP': <Action.UP: 2>, 'RIGHT': <Action.RIGHT: 3>, 'LEFT': <Action.LEFT: 4>, 'DOWN': <Action.DOWN: 5>, 'UPRIGHT': <Action.UPRIGHT: 6>, 'UPLEFT': <Action.UPLEFT: 7>, 'DOWNRIGHT': <Action.DOWNRIGHT: 8>, 'DOWNLEFT': <Action.DOWNLEFT: 9>, 'UPFIRE': <Action.UPFIRE: 10>, 'RIGHTFIRE': <Action.RIGHTFIRE: 11>, 'LEFTFIRE': <Action.LEFTFIRE: 12>, 'DOWNFIRE': <Action.DOWNFIRE: 13>, 'UPRIGHTFIRE': <Action.UPRIGHTFIRE: 14>, 'UPLEFTFIRE': <Action.UPLEFTFIRE: 15>, 'DOWNRIGHTFIRE': <Action.DOWNRIGHTFIRE: 16>, 'DOWNLEFTFIRE': <Action.DOWNLEFTFIRE: 17>}
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
    def act(self, action: Action) -> int: ...
    def actContinuous(self, r: float, theta: float, fire: float) -> int: ...
    @overload
    def act(self, action: int) -> int: ...
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
