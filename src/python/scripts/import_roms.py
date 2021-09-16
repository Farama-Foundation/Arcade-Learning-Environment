import os
import argparse
import pathlib
import operator
import warnings

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


def import_roms(romdir, datadir, pkg=None, dry_run=False):
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
        identifier = str(path) if pkg is None else f"{pkg}/{path.name}"
        if not dry_run:
            copyfile(path, datadir / f"{rom}.bin")
        print(f"\033[92m{'[SUPPORTED]': <15}\033[0m {rom: >20} {identifier: >30}")

    print("\n")
    # Print unsuported
    for path in unsupported:
        identifier = str(path) if pkg is None else f"{pkg}/{path.name}"
        print(f"\033[91m{'[NOT SUPPORTED]': <15}\033[0m {'': >20} {identifier: >30}")

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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--import-from-pkg")
    group.add_argument("romdir", help="Directory containing ROMs", nargs="?")

    args = parser.parse_args()

    if args.romdir:
        romdir = pathlib.Path(args.romdir)

        if not romdir.exists():
            print(f"Path {romdir} doesn't exist.")
            exit(1)
    elif args.import_from_pkg:
        root, subpackage = args.import_from_pkg.split(".", maxsplit=1)
        try:
            with resources.path(root, subpackage) as path:
                romdir = path.resolve()
        except ModuleNotFoundError:
            print(f"Unable to find module {root}.")
            exit(1)
        except Exception as e:
            print(f"Unknown error {str(e)}.")
            exit(1)
        finally:
            if not romdir.exists():
                print(f"Unable to find path {subpackage} in module {root}.")
                exit(1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="ale_py.roms"
        )

        datadir = resources.files("ale_py.roms")
        import_roms(romdir, datadir, pkg=args.import_from_pkg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
