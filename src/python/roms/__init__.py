import os
import warnings
import pathlib

from typing import List, Union, Dict

from ale_py.roms.utils import (
    SupportedPackage as _SupportedPackage,
    SupportedPlugin as _SupportedPlugin,
)


# Precedence is as follows:
#  1. Internal ROMs
#  2. External ROMs
#  3. ROMs from atari-py.roms
#  4. ROMs from atari-py-roms.roms
_SUPPORTED_PACKAGES: List[Union[_SupportedPackage, _SupportedPlugin]] = [
    _SupportedPackage("ale_py.roms"),
    _SupportedPlugin("ale_py.roms"),
    _SupportedPackage("atari_py.atari_roms"),
    _SupportedPackage("atari_py_roms.atari_roms"),
]


def _resolve_roms() -> List[str]:
    roms: Dict[str, pathlib.Path] = {}
    for package in _SUPPORTED_PACKAGES:

        try:
            supported, unsupported = package.resolve()

            if isinstance(package, _SupportedPackage) and package.package.startswith(
                "atari_py"
            ):
                warnings.warn(
                    "Importing atari-py roms won't be supported in future releases of ale-py.",
                    category=DeprecationWarning,
                    stacklevel=3,
                )
            if len(unsupported) > 0:
                parent = unsupported[0].parent
                names = ", ".join(map(lambda path: path.name, unsupported))
                warnings.warn(
                    f"{package} contains unsupported ROMs: {parent}{os.sep}{{{names}}}"
                )

            roms.update(supported)
        except ModuleNotFoundError:
            pass

    globals().update(roms)
    return list(roms.keys())


__all__ = _resolve_roms()
