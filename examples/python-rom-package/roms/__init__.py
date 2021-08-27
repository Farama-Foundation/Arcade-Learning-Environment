import sys
import pathlib

if sys.version_info < (3, 8):
    import importlib_resources as resources
else:
    import importlib.resources as resources

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
