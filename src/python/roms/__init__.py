from __future__ import annotations

import base64
import hashlib
import json
import tarfile
import warnings
from pathlib import Path

import requests


# Extract each valid ROM into each dir in installation_dirs
def _download_roms():
    # this is a map of {rom.bin : md5 checksum}
    all_roms = json.load(open(Path(__file__).parent / "md5.json"))

    # use requests to download the base64 file
    url = "https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64"
    r = requests.get(url, allow_redirects=False)

    # decode the b64 into the tar.gz and save it
    tar_gz = base64.b64decode(r.content)
    save_path = Path(__file__).parent / "Roms.tar.gz"
    open(save_path, "wb").write(tar_gz)

    # iterate through each file in the tar
    tar_fp = tarfile.open(name=save_path)
    for member in tar_fp.getmembers():
        # ignore if this is not a valid bin file
        if not (member.isfile() and member.name.endswith(".bin")):
            continue

        # grab the rom name from the member name
        # by default, the roms have the hierarchy ROM/rom_name/rom_name.bin
        rom_name = member.name.split("/")[-1]

        # assert that this member.name should be supported
        assert (
            rom_name in all_roms
        ), f"File {rom_name} not supported. Please report this to a dev."

        # extract the rom file from the archive
        rom_bytes = tar_fp.extractfile(member).read()  # pyright: ignore[reportOptionalMemberAccess]

        # get hash
        md5 = hashlib.md5()
        md5.update(rom_bytes)
        md5_hash = md5.hexdigest()

        # assert that the hash matches
        assert (
            md5_hash == all_roms[rom_name]
        ), f"Rom {rom_name}'s hash does not match what was expected. Please report this to a dev."

        # save this rom
        rom_path = Path(__file__).parent / rom_name
        open(rom_path, "wb").write(rom_bytes)

        print(f"Downloaded and unpacked {rom_name}.")


def get_rom_path(name: str) -> Path | None:
    """Expects name as a snake_case name, returns the full path of the .bin file if it's valid, otherwise returns None."""
    # the theoretical location of the binary rom file
    bin_file = f"{name}.bin"
    bin_path = Path(__file__).parent / bin_file

    # check if it exists within the md5.json
    all_roms = json.load(open(Path(__file__).parent / "md5.json"))
    if bin_file not in all_roms:
        warnings.warn(f"Rom {name} not supported.")
        return None

    if bin_path.exists():
        # if the path exists, just return it
        return bin_path
    else:
        # if it doesn't exist, we need to install the roms, then return it
        print(f"Could not find rom {name}, downloading roms to device...")
        _download_roms()
        return bin_path


def get_all_rom_ids() -> list[str]:
    """Returns a list of all available rom_ids, ie: ['tetris', 'pong', 'zaxxon', ...]."""
    all_roms = json.load(open(Path(__file__).parent / "md5.json"))
    return [key.split(".")[0] for key in all_roms.keys()]
