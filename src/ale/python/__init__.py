"""Python module for interacting with ALE c++ interface and gymnasium wrapper."""

import os
import platform
import sys
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
    # The best approach seems to be using LoadLibraryEx
    # with user defined search paths. This kind of acts like
    # $ORIGIN or @loader_path on Unix / macOS.
    # This way we guarantee we load OUR DLLs.
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        os.add_dll_directory(packagedir)
    else:
        # TODO: Py38: Remove AddDllDirectory
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        if hasattr(kernel32, "AddDllDirectory"):
            kernel32.AddDllDirectory(packagedir)

# version string from package.
import importlib.metadata as metadata  # noqa: E402

__version__ = metadata.version(__package__)

# Import native shared library
from ale_py._ale_py import (  # noqa: E402
    SDL_SUPPORT,
    Action,
    ALEInterface,
    ALEState,
    LoggerMode,
)

__all__ = ["Action", "ALEInterface", "ALEState", "LoggerMode", "SDL_SUPPORT"]


try:
    from ale_py.env import AtariEnv, AtariEnvStepMetadata
    from ale_py.vector_env import VectorAtariEnv

    __all__ += ["AtariEnv", "AtariEnvStepMetadata", "VectorAtariEnv"]

    from ale_py.registration import register_v0_v4_envs, register_v5_envs

    register_v0_v4_envs()
    register_v5_envs()
except ImportError:
    pass
