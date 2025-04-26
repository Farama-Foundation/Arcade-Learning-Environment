"""Python module for interacting with ALE c++ interface and gymnasium wrapper."""

import importlib.metadata as metadata
import os
import platform
import warnings

packagedir = os.path.abspath(os.path.dirname(__file__))

# Make sure to adjust the filter to show DeprecationWarning
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)

if platform.system() == "Windows":
    try:
        import ctypes

        ctypes.CDLL("vcruntime140.dll")
        ctypes.CDLL("msvcp140.dll")
    except OSError:
        raise OSError(
            """Microsoft Visual C++ Redistribution Pack is not installed.
It can be downloaded from https://aka.ms/vs/16/release/vc_redist.x64.exe."""
        )

    # Loading DLLs on Windows is kind of a disaster
    # The best approach seems to be using LoadLibraryEx with user defined search paths.
    # This kind of acts like $ORIGIN or @loader_path on Unix / macOS.
    # This way we guarantee we load OUR DLLs.
    os.add_dll_directory(packagedir)

__version__ = metadata.version(__package__)

# Import native shared library
from ale_py._ale_py import SDL_SUPPORT, Action, ALEInterface, ALEState, LoggerMode

__all__ = ["Action", "ALEInterface", "ALEState", "LoggerMode", "SDL_SUPPORT"]


try:
    # As the vector interface is an optional cmake build, it's not assumed to exist
    from ale_py._ale_py import ALEVectorInterface

    __all__ += ["ALEVectorInterface"]
except ImportError:
    pass


from ale_py.env import AtariEnv, AtariEnvStepMetadata
from ale_py.vector_env import AtariVectorEnv

__all__ += ["AtariEnv", "AtariEnvStepMetadata", "AtariVectorEnv"]

from ale_py.registration import register_v0_v4_envs, register_v5_envs

register_v0_v4_envs()
register_v5_envs()
