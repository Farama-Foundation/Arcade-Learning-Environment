import argparse
import pathlib
import shutil
import warnings
from typing import Optional

import ale_py
import importlib_resources as resources


def import_roms(
    romdir: pathlib.Path,
    datadir: pathlib.Path,
    pkg: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Recursively copies all compatible ROMs in romdir
    to datadir using the proper filename for the ALE.
    """

    supported = {}
    unsupported = []
    for path in romdir.glob("**/*.bin"):
        rom = ale_py.ALEInterface.isSupportedROM(path)
        if rom is not None:
            supported[rom] = path
        else:
            unsupported.append(path)

    # Copy over supported files
    for rom, path in supported.items():
        identifier = str(path) if pkg is None else f"{pkg}/{path.name}"
        if not dry_run:
            shutil.copyfile(path, datadir / f"{rom}.bin")
        print(f"\033[92m{'[SUPPORTED]': <15}\033[0m {rom: >20} {identifier: >30}")

    print("\n")
    # Print unsuported
    for path in unsupported:
        identifier = str(path) if pkg is None else f"{pkg}/{path.name}"
        print(f"\033[91m{'[NOT SUPPORTED]': <15}\033[0m {'': >20} {identifier: >30}")

    # Print summary
    if not dry_run:
        print(f"\nImported {len(supported)} / {len(supported)} ROMs")


def main() -> None:
    """
    CLI for ale-import-roms
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=ale_py.__version__)
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
        if "." in args.import_from_pkg:
            root, subpackage = args.import_from_pkg.split(".", maxsplit=1)
        else:
            root, subpackage = args.import_from_pkg, None
        try:
            with resources.path(root, subpackage) as path:
                romdir = path.resolve()
                if not romdir.exists():
                    print(f"Unable to find path {subpackage} in module {root}.")
                    exit(1)
        except ModuleNotFoundError:
            print(f"Unable to find module {root}.")
            exit(1)
        except Exception as e:
            print(f"Unknown error {str(e)}.")
            exit(1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="ale_py.roms"
        )

        datadir = resources.files("ale_py.roms")
        import_roms(romdir, datadir, pkg=args.import_from_pkg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
