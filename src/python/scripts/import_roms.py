import os
import argparse
import pathlib
import operator

import importlib_resources as resources

from shutil import copyfile

from ale_py import __version__, ALEInterface


def scantree(dir, recurse=True):
    """
    Recursive os.scandir
    """
    with os.scandir(dir) as root:
        for entry in root:
            if recurse and entry.is_dir(follow_symlinks=False):
                yield from scantree(entry.path)
            elif entry.is_file():
                yield pathlib.Path(entry)


def import_roms(romdir, datadir, dry_run=False):
    """
    Recursively copies all compatible ROMs in romdir
    to datadir using the proper filename for the ALE.
    """
    # Get all files in romdir ending in '.bin'
    entries = list(filter(lambda file: file.suffix == ".bin", scantree(romdir)))
    # Get ROM id or None from isSupportedROM
    ids = list(map(ALEInterface.isSupportedROM, entries))
    # Filter only the supported ROMs
    supported = list(filter(lambda args: args[0] is not None, zip(ids, entries)))
    # Get the set of ROMs that are unsupported
    unsupported = set(entries) - set(map(operator.itemgetter(1), supported))

    # Copy over supported files
    for rom, path in supported:
        if not dry_run:
            copyfile(path, datadir / f"{rom}.bin")
        print(f"\033[92m{'[SUPPORTED]': <15}\033[0m {rom: >20} {str(path): >30}")

    print("\n")
    # Print unsuported
    for path in unsupported:
        print(f"\033[91m{'[NOT SUPPORTED]': <15}\033[0m {'': >20} {str(path): >30}")

    # Print summary
    if not dry_run:
        print(f"\nImported {len(supported)} / {len(entries)} ROMs")


def main():
    """
    CLI for ale-import-roms
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("romdir", help="Directory containing ROMs")
    args = parser.parse_args()

    romdir = pathlib.Path(args.romdir)
    datadir = resources.files("ale_py.roms")

    import_roms(romdir, datadir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
