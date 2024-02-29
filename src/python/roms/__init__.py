from __future__ import annotations

import base64
import functools
import hashlib
import json
import tarfile
import warnings
from pathlib import Path

import requests


@functools.lru_cache(maxsize=1)
def _get_all_rom_hashes() -> dict[str, str]:
    # this is a map of {rom.bin : md5 checksum}
    with open(Path(__file__).parent / "md5.json") as f:
        return json.load(f)


def _download_roms() -> None:
    """Unpacks all roms from the tar.gz file, then matches it to the expected md5 checksum."""
    all_roms = _get_all_rom_hashes()

    # load the b64 file
    if (Path(__file__).parent / "Roms.tar.gz.b64").exists:
        with open(Path(__file__).parent / "Roms.tar.gz.b64", "r") as f:
            tar_gz_b64 = f.read()
    else:
        # fallback to plain url download in case something went wrong during build publish
        url = "https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64"
        tar_gz_b64 = requests.get(url, allow_redirects=False).content

    # decode the b64 into the tar.gz and save it
    tar_gz = base64.b64decode(tar_gz_b64)
    save_path = Path(__file__).parent / "Roms.tar.gz"
    with open(save_path, "wb") as f:
        f.write(tar_gz)

    # iterate through each file in the tar
    with tarfile.open(name=save_path) as tar_fp:
        for member in tar_fp.getmembers():
            # ignore if this is not a valid bin file
            if not (member.isfile() and member.name.endswith(".bin")):
                continue

            # grab the rom name from the member name, ie: `pong.bin`
            # member.name is ROM/rom_name/rom_name.bin, so we just want the last thing
            rom_name = member.name.split("/")[-1]

            # assert that this member.name should be supported
            assert (
                rom_name in all_roms
            ), f"File {rom_name} not supported. Please report this to a dev."

            # extract the rom file from the archive
            rom_bytes = tar_fp.extractfile(
                member
            ).read()  # pyright: ignore[reportOptionalMemberAccess]

            # get hash
            md5 = hashlib.md5()
            md5.update(rom_bytes)
            md5_hash = md5.hexdigest()

            # assert that the hash matches
            assert md5_hash == all_roms[rom_name], (
                f"Rom {rom_name}'s hash ({md5_hash}) does not match what was expected ({all_roms[rom_name]}). "
                "Please report this to a dev."
            )

            # save this rom
            rom_path = Path(__file__).parent / rom_name
            with open(rom_path, "wb") as rom_fp:
                rom_fp.write(rom_bytes)


def get_rom_path(name: str) -> Path | None:
    """Expects name as a snake_case name, returns the full path of the .bin file if it's valid, otherwise returns None."""
    # the theoretical location of the binary rom file
    bin_file = f"{name}.bin"
    bin_path = Path(__file__).parent / bin_file

    # check if it exists within the the hash dictionary
    all_roms = _get_all_rom_hashes()
    if bin_file not in all_roms:
        warnings.warn(f"Rom {name} not supported.")
        return None

    # if the bin_path doesn't exist, TELL SOMEONE PANIC THE WORLD IS ENDING
    assert (
        bin_path.exists()
    ), f"Could not find the rom at {bin_path}, seems like rom download has gone wrong. Please let a dev know."

    # return the path
    return bin_path


def get_all_rom_ids() -> list[str]:
    """Returns a list of all available rom_ids, ie: ['tetris', 'pong', 'zaxxon', ...]."""
    all_roms = _get_all_rom_hashes()
    return [key.split(".")[0] for key in all_roms.keys()]
