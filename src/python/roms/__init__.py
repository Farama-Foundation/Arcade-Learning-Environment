import functools
import operator
import os
import pathlib
import warnings
from typing import Dict, List

from ale_py.roms import utils

# Precedence is as follows:
#  1. Internal ROMs
#  2. External ROMs
#  3. ROMs from atari-py.roms
#  4. ROMs from atari-py-roms.roms
_ROM_PLUGINS: List[utils.RomPlugin] = [
    utils.SupportedPackage("ale_py.roms"),
    utils.SupportedEntryPoint("ale_py.roms"),
    utils.SupportedPackage("atari_py.atari_roms"),
    utils.SupportedPackage("atari_py_roms.atari_roms"),
]

# Environment variable for ROM discovery.
# ale-py will search for supported ROMs in:
#   ${ALE_PY_ROM_DIR}/*.bin
_ROM_DIRECTORY_ENV_KEY = "ALE_PY_ROM_DIR"
_ROM_DIRECTORY_ENV_VALUE = os.environ.get(_ROM_DIRECTORY_ENV_KEY, None)

if _ROM_DIRECTORY_ENV_VALUE is not None:
    _ROM_PLUGINS.append(utils.SupportedDirectory(_ROM_DIRECTORY_ENV_VALUE))


@functools.lru_cache(maxsize=None)
def _resolve_roms() -> Dict[str, pathlib.Path]:
    global _ROM_PLUGINS

    roms: Dict[str, pathlib.Path] = {}
    for package in _ROM_PLUGINS:

        try:
            # Resolve supported / unsupported roms
            supported, unsupported = package.resolve()

            # We'll now get the update delta. The reason for this is two fold:
            #     1) We should only display atari-py deprecation when it would have
            #        imported ROMs.
            #     2) ROM priority holds. When you import ROMs they'll all come from
            #        a single source of truth.
            #
            roms_delta_keys = list(
                filter(lambda rom: rom not in roms, supported.keys())
            )
            roms_delta = dict(
                zip(
                    roms_delta_keys,
                    map(
                        functools.partial(operator.getitem, supported), roms_delta_keys
                    ),
                )
            )

            if (
                isinstance(package, utils.SupportedPackage)
                and package.package.startswith("atari_py")
                and len(roms_delta) > 0
            ):
                warnings.warn(
                    "Automatic importing of atari-py roms won't be supported in future releases of ale-py. "
                    "Please migrate over to using `ale-import-roms` OR an ALE-supported ROM package. "
                    "To make this warning disappear you can run "
                    f"`ale-import-roms --import-from-pkg {package.package}`."
                    "For more information see: https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
            if len(unsupported) > 0:
                parent = unsupported[0].parent
                names = ", ".join(map(lambda path: path.name, unsupported))
                warnings.warn(
                    f"{package} contains unsupported ROMs: {parent}{os.sep}{{{names}}}",
                    category=ImportWarning,
                    stacklevel=2,
                )

            roms.update(roms_delta)
        except ModuleNotFoundError:
            pass

    return roms


_ROMS = None


def __dir__() -> List[str]:
    global _ROMS

    if _ROMS is None:
        _ROMS = _resolve_roms()
    return list(_ROMS.keys())


def __getattr__(name: str) -> pathlib.Path:
    global _ROMS

    if _ROMS is None:
        _ROMS = _resolve_roms()
    if name not in _ROMS:
        raise AttributeError(
            f"ROM {name} not found. Available ROMs: {','.join(__dir__())}."
        )

    return _ROMS[name]
