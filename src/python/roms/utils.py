import abc
import pathlib
import sys
from typing import Callable, Dict, List, Tuple, Union

# TODO: importlib.resources.files has a bug that raises
# a TypeError when there's no __init__.py file in the package dir.
# importlib_resources doesn't suffer from this bug.
import importlib_resources as resources

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    import importlib.metadata as metadata

from ale_py._ale_py import ALEInterface


def rom_id_to_name(rom) -> str:
    """
    Let the ROM ID be the ROM identifier in snakecase.
        For example, `space_invaders`
    The ROM name is the ROM ID in camelcase.
        For example, `SpaceInvaders`

    This function converts the ROM ID to the ROM name.
        i.e., snakecase -> camelcase
    """
    return rom.title().replace("_", "")


def rom_name_to_id(rom) -> str:
    """
    Let the ROM ID be the ROM identifier in snakecase.
        For example, `space_invaders`
    The ROM name is the ROM ID in camelcase.
        For example, `SpaceInvaders`

    This function converts the ROM name to the ROM ID.
        i.e., camelcase -> snakecase
    """
    return "".join(
        map(lambda ch: "_" + ch.lower() if ch.isupper() else ch, rom)
    ).lstrip("_")


class RomPlugin(abc.ABC):
    @abc.abstractmethod
    def resolve(self) -> Tuple[Dict[str, pathlib.Path], List[pathlib.Path]]:
        """
        Resolve the ROMs supported by this plugin.
        """
        pass


class SupportedPackage(RomPlugin):
    def __init__(self, package: str):
        self.package = package

    def resolve(self) -> Tuple[Dict[str, pathlib.Path], List[pathlib.Path]]:
        roms: Dict[str, pathlib.Path] = {}
        unsupported: List[pathlib.Path] = []

        # Iterate over all ROMs in the specified package
        for resource in filter(
            lambda file: file.suffix == ".bin", resources.files(self.package).iterdir()
        ):
            resolved = resource.resolve()
            rom = ALEInterface.isSupportedROM(resolved)
            # If the ROM is supported we normalize the name and add it to
            # the dictionary of ROMs
            if rom is not None:
                roms[rom_id_to_name(rom)] = resource.resolve()
            else:
                unsupported.append(resolved)

        return roms, unsupported

    def __str__(self) -> str:
        return f"{self.package}"

    def __repr__(self) -> str:
        return f"SupportedPackage[{self.package}]"


class SupportedEntryPoint(RomPlugin):
    def __init__(self, group: str):
        self.group = group

    def resolve(self) -> Tuple[Dict[str, pathlib.Path], List[pathlib.Path]]:
        roms: Dict[str, pathlib.Path] = {}
        unsupported: List[pathlib.Path] = []

        # Iterate over all entrypoints in this group
        for external in metadata.entry_points(group=self.group):
            # We load the external load ROM function and
            # update the ROM dict with the result
            external_fn: Callable[[], List[Union[pathlib.Path, str]]] = external.load()
            for path in external_fn():
                path = pathlib.Path(path)
                rom = ALEInterface.isSupportedROM(path)
                if rom is not None:
                    roms[rom_id_to_name(rom)] = path
                else:
                    unsupported.append(path)

        return roms, unsupported

    def __str__(self) -> str:
        return f"{self.group}"

    def __repr__(self) -> str:
        return f"SupportedEntryPoint[{self.group}]"


class SupportedDirectory(RomPlugin):
    def __init__(self, directory: Union[str, pathlib.Path]):
        self.directory = pathlib.Path(directory)

    def resolve(self) -> Tuple[Dict[str, pathlib.Path], List[pathlib.Path]]:
        roms: Dict[str, pathlib.Path] = {}
        unsupported: List[pathlib.Path] = []

        # Iterate over all bin files in directory
        for path in self.directory.glob("*.bin"):
            rom = ALEInterface.isSupportedROM(path)
            if rom is not None:
                roms[rom_id_to_name(rom)] = path
            else:
                unsupported.append(path)

        return roms, unsupported

    def __str__(self) -> str:
        return f"{self.directory}"

    def __repr__(self) -> str:
        return f"SupportedDirectory[{self.directory}]"
