import functools
import json
import pathlib
import re
from os import path

from ale_py import ALEInterface


def rom_id_to_name(rom: str) -> str:
    """
    Let the ROM ID be the ROM identifier in snakecase.
        For example, `space_invaders`
    The ROM name is the ROM ID in camelcase.
        For example, `SpaceInvaders`

    This function converts the ROM ID to the ROM name.
        i.e., snakecase -> camelcase
    """
    return rom.title().replace("_", "")


def rom_name_to_id(rom: str) -> str:
    """
    Let the ROM ID be the ROM identifier in snakecase.
        For example, `space_invaders`
    The ROM name is the ROM ID in camelcase.
        For example, `SpaceInvaders`

    This function converts the ROM name to the ROM ID.
        i.e., camelcase -> snakecase
    """
    name_to_id_re = re.compile(r"([0-9]*[A-Z][a-z]*(\d*$)?)")
    return name_to_id_re.sub(r"\1_", rom).lower().rstrip("_")


@functools.lru_cache(maxsize=None)
def __getattr__(rom_name: str) -> pathlib.Path:
    """Return the path to a ROM."""
    # get list of available ROMs from the md5 file
    # realistically we don't need the MD5s since PyPI will perform hash checks for us
    base_path = path.dirname(__file__)
    rom_names = [
        rom_id_to_name(n.split(".")[0])
        for n in json.load(open(path.join(base_path, "md5.json"))).keys()
    ]

    # check that the rom is valid
    if rom_name not in rom_names:
        raise AttributeError(
            f"No ROM named {rom_name}. Supported ROMs: {', '.join(rom_names)}"
        )

    # return it as a pathlib object
    return pathlib.Path(path.join(base_path, f"{rom_name_to_id(rom_name)}.bin"))
