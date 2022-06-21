import abc
import inspect
import pathlib
import sys
import warnings
from typing import Optional, Union

from ale_py.roms import utils

# pylint: disable=g-import-not-at-top
if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    import importlib.metadata as metadata

if sys.version_info >= (3, 9):
    import importlib.resources as resources
else:
    import importlib_resources as resources
# pylint: enable=g-import-not-at-top

from ale_py import ALEInterface


class Plugin(abc.ABC):
    @abc.abstractmethod
    def resolve(self, id: str) -> Optional[pathlib.Path]:
        """
        Resolve the ROMs supported by this plugin.
        """
        pass


class Package(Plugin):
    def __init__(self, package: str):
        self.package = package

    def resolve(self, id: str) -> Optional[pathlib.Path]:
        rom = resources.files(self.package).joinpath(f"{id}.bin")
        if not rom.exists():
            return None
        return rom

    def __str__(self) -> str:
        return f"{self.package}"

    def __repr__(self) -> str:
        return f"Package[{self.package}]"


class EntryPoint(Plugin):
    def __init__(self, group: str):
        self.group = group

    def resolve(self, id: str) -> Optional[pathlib.Path]:
        # Iterate over all entrypoints in this group
        for external in metadata.entry_points(group=self.group):
            # We load the external load ROM function and
            # update the ROM dict with the result
            external_fn = external.load()
            sig = inspect.signature(external_fn)

            # Check signature of entry-point for backwards compatibility
            if not sig.parameters:
                rom = None
                for path in external_fn():
                    if path.stem == id:
                        rom = path
                        break
            elif len(sig.parameters) == 1:
                rom = external_fn(id)
            else:
                warnings.warn(
                    f"Entry point {external_fn} has an invalid signature "
                    f"with parameters {sig.parameters}"
                )
                continue

            if rom is None:
                continue
            # We'll manually check if the ROM is supported because we don't
            # want to exit early in case one plugin provided an invalid ROM.
            if ALEInterface.isSupportedROM(rom):
                return rom
        return None

    def __str__(self) -> str:
        return f"{self.group}"

    def __repr__(self) -> str:
        return f"EntryPoint[{self.group}]"


class Directory(Plugin):
    def __init__(self, directory: Union[str, pathlib.Path]):
        self.directory = pathlib.Path(directory)

    def resolve(self, id: str) -> Optional[pathlib.Path]:
        rom = self.directory.joinpath(f"{id}.bin")
        if not rom.exists():
            return None
        return rom

    def __str__(self) -> str:
        return f"{self.directory}"

    def __repr__(self) -> str:
        return f"Directory[{self.directory}]"
