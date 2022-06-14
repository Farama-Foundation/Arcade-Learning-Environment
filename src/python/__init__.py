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

# TODO Py38: Once 3.7 is deprecated use importlib.metadata to parse
# version string from package.
try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata
try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = "unknown"

# Import native shared library
from ale_py._ale_py import SDL_SUPPORT, Action, ALEInterface, ALEState, LoggerMode

__all__ = ["Action", "ALEInterface", "ALEState", "LoggerMode", "SDL_SUPPORT"]
