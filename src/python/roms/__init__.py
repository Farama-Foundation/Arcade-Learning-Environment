import functools
import json
import pathlib
from os import path


@functools.lru_cache(maxsize=None)
def __getattr__(rom_name: str) -> pathlib.Path:
    """Return the path to a ROM."""
    # get list of available ROMs from the md5 file
    # realistically we don't need the MD5s since PyPI will perform hash checks for us
    base_path = path.dirname(__file__)
    rom_names = [
        n.split(".")[0]
        for n in json.load(open(path.join(base_path, "md5.json"))).keys()
    ]

    # check that the rom is valid
    if rom_name not in rom_names:
        raise AttributeError(
            f"No ROM named {rom_name}. Supported ROMs: {', '.join(rom_names)}"
        )

    # return it as a pathlib object
    return pathlib.Path(path.join(base_path, f"{rom_name}.bin"))
