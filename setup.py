"""Setup file for ALE."""

import os
import re

def parse_version(version_file):
    """Parse version from `version_file`.

    If we're running on CI, i.e., CIBUILDWHEEL is set, then we'll parse
    the version from `GITHUB_REF` using the official semver regex.

    If we're not running in CI we'll append the current git SHA to the
    version identifier.
    """
    semver_regex = r"(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?"
    semver_prog = re.compile(semver_regex)

    with open(version_file) as fp:
        version = fp.read().strip()
        assert semver_prog.match(version) is not None

    return version

def get_version():
    """Get the version for the package."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    version_file = os.path.join(current_dir, "version.txt")
    return parse_version(version_file)

__version__ = get_version()
