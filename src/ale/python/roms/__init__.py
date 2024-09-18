"""Rom module with functions for collecting individual and all ROMS files."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import warnings
from pathlib import Path


@functools.lru_cache(maxsize=1)
def _get_expected_bin_hashes() -> dict[str, str]:
    # this is a map of {rom.bin : md5 checksum}
    with open(Path(__file__).parent / "md5.json") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def get_all_rom_ids() -> list[str]:
    """Returns a list of all available rom_ids, ie: ['tetris', 'pong', 'zaxxon', ...]."""
    return [key.split(".")[0] for key in _get_expected_bin_hashes().keys()]


def get_rom_path(name: str) -> Path | None:
    """Expects name as a snake_case name, returns the full path of the .bin file if it's valid, otherwise returns None."""
    # grab the roms_dir environment
    if os_environ_path := os.environ.get("ALE_ROMS_DIR"):
        roms_dir = Path(os_environ_path)
        print(f"Loading roms from {roms_dir.absolute()}...")
    else:
        roms_dir = Path(__file__).parent

    # make sure the roms dir exists
    if not roms_dir.exists():
        raise NotADirectoryError(f"ROM directory {roms_dir.absolute()} doesn't exist")
    elif not roms_dir.is_dir():
        raise NotADirectoryError(
            f"ROM directory {roms_dir.absolute()} isn't a directory"
        )

    # the theoretical location of the binary rom file
    bin_file = f"{name}.bin"
    bin_path = roms_dir / bin_file

    # check if it exists within the the hash dictionary
    bin_hashes = _get_expected_bin_hashes()
    if bin_file not in bin_hashes.keys():
        warnings.warn(f"Rom {name} not supported.")
        return None

    # check the rom hash
    with open(bin_path, "rb") as bin_fp:
        md5 = hashlib.md5()
        md5.update(bin_fp.read())
        md5_hash = md5.hexdigest()

        if md5_hash != bin_hashes[bin_file]:
            raise OSError(
                f"The hash of {bin_file} does not match what was expected!\n"
                f"Expected ({bin_hashes[bin_file]}),\n"
                f"Obtained ({md5_hash})."
            )

    # return the path
    return bin_path
