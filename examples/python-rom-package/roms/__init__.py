import pathlib
import sys

if sys.version_info >= (3, 9):
    import importlib.resources as resources
else:
    import importlib_resources as resources

from typing import List


def atari57() -> List[pathlib.Path]:
    return list(
        map(
            pathlib.Path,
            filter(
                lambda file: file.suffix == ".bin", resources.files(__name__).iterdir()
            ),
        )
    )
