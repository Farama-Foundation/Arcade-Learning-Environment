import functools
import logging
import os
import pathlib
import sys
import warnings
from typing import List, Optional

from ale_py import ALEInterface
from ale_py.roms import plugins, utils

# pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 9):
    import importlib.resources as resources
else:
    import importlib_resources as resources
# pylint: enable=g-import-not-at-top

# Precedence is as follows:
#  1. Internal ROMs
#  2. External ROMs
#  3. ROMs from atari-py.roms
#  4. ROMs from atari-py-roms.roms
_ROM_PLUGIN_REGISTRY: List[plugins.Plugin] = [
    plugins.Package("ale_py.roms"),
    plugins.EntryPoint("ale_py.roms"),
    plugins.Package("atari_py.atari_roms"),
    plugins.Package("atari_py_roms.atari_roms"),
]

# Environment variable for ROM discovery.
# ale-py will search for supported ROMs in:
#   ${ALE_PY_ROM_DIR}/*.bin
_ROM_DIRECTORY_ENV_KEY = "ALE_PY_ROM_DIR"
_ROM_DIRECTORY_ENV_VALUE = os.environ.get(_ROM_DIRECTORY_ENV_KEY, None)

if _ROM_DIRECTORY_ENV_VALUE is not None:
    _ROM_PLUGIN_REGISTRY.append(plugins.Directory(_ROM_DIRECTORY_ENV_VALUE))


def _resolve_rom(name: str) -> Optional[pathlib.Path]:
    """Resolve a ROM path from the ROM registry."""
    for package in _ROM_PLUGIN_REGISTRY:
        rom_id = utils.rom_name_to_id(name)

        # Resolve ROM from package
        try:
            rom_path = package.resolve(rom_id)
        except ModuleNotFoundError:
            continue

        # Failed to resolve ROM
        if rom_path is None:
            logging.debug(f"{package} did not resolve {rom_id}.")
            continue

        # ROM isn't supported
        resolved_id = ALEInterface.isSupportedROM(rom_path)
        if resolved_id is None:
            warnings.warn(
                f"{package} contains unsupported ROM: {rom_path}",
                category=ImportWarning,
                stacklevel=2,
            )
            continue

        # If the ROMs resolved differently
        if resolved_id != rom_id:
            warnings.warn(
                f"{package} contains ROM {rom_id} which doesn't resolve to {resolved_id}. "
                "This is most likely caused by a filename mismatch.",
            )
            continue

        # Deprecation warning for atari-py
        if isinstance(package, plugins.Package) and package.package.startswith(
            "atari_py"
        ):
            warnings.warn(
                "Automatic importing of atari-py roms won't be supported in future releases of ale-py. "
                "Please migrate over to using `ale-import-roms` OR an ALE-supported ROM package. "
                "To make this warning disappear you can run "
                f"`ale-import-roms --import-from-pkg {package.package}`. "
                "For more information see: https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management",
                category=DeprecationWarning,
                stacklevel=2,
            )

        return rom_path
    # Return None if we couldn't resolve the ROM from any package
    return None


@functools.lru_cache(maxsize=None)
def __dir__() -> List[str]:
    """Return the ROM directory, i.e., a list of all supported ROMs."""
    md5s = resources.files(__name__).joinpath("md5.txt")
    if not md5s.exists():
        raise FileNotFoundError(
            f"ROM md5 resource couldn't be found. "
            "Are you running from a development environment? "
        )
    with md5s.open() as fp:
        lines = filter(
            lambda line: line.strip() and not line.startswith("#"), fp.readlines()
        )
        roms = [pathlib.Path(rom) for _, rom in map(str.split, lines)]
        return [utils.rom_id_to_name(rom.stem) for rom in roms]


@functools.lru_cache(maxsize=None)
def __getattr__(name: str) -> pathlib.Path:
    """Return the path to a ROM."""
    roms = __dir__()
    if name not in roms:
        raise AttributeError(f"No ROM named {name}. Supported ROMs: {', '.join(roms)}")

    path = _resolve_rom(name)
    if path is None:
        raise AttributeError(
            f"Failed to resolve ROM `{name}` from plugins "
            f"{', '.join(map(lambda plugin: f'`{repr(plugin)}`', _ROM_PLUGIN_REGISTRY))}. "
            "If you own a license to use the necessary ROMs for research purposes you can download them "
            f"via `pip install autorom[accept-rom-license]`. Otherwise, you should try importing `{name}` "
            f"via the command `ale-import-roms`. If you believe this is a mistake perhaps your copy of `{name}` "
            "is unsupported. To check if this is the case try providing the environment variable "
            "`PYTHONWARNINGS=default::ImportWarning:ale_py.roms`. For more information see: "
            "https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management"
        )
    return path


def register_plugin(plugin: plugins.Plugin, *, index: int = 0) -> None:
    """Register a ROM plugin."""
    if not issubclass(type(plugin), plugins.Plugin):
        raise ValueError(f"{repr(plugin)} is not a valid ROM plugin.")
    _ROM_PLUGIN_REGISTRY.insert(index, plugin)
    __getattr__.cache_clear()


__all__ = ["register_plugin"] + __dir__()
