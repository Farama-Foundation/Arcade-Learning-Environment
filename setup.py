from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
import shlex
import re


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", config=[]):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.config = config


class CMakeBuild(build_ext):
    def build_extensions(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the extensions: %s"
                % ", ".join(ext.name for ext in self.extensions)
            )

        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = "Debug" if self.debug else "Release"

            if sys.platform.startswith("linux"):
                ext_suffix = ".so"
            elif sys.platform.startswith("darwin"):
                ext_suffix = ".dylib"
            elif sys.platform.startswith("win"):
                ext_suffix = ".dll"
            else:
                raise RuntimeError(
                    'CMakeBuild: Platform "%s" not recognized' % sys.platform
                )

            cmake_build_args = []
            cmake_config_args = [
                "-DOUTPUT_NAME={}".format(ext.name),
                "-DCMAKE_BUILD_TYPE={}".format(cfg),
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), self.build_temp
                ),
                "-DPYTHON_MODULE_EXTENSION={}".format(ext_suffix),
            ] + ext.config

            # -DCMAKE_BUILD_TYPE doesn't work on Windows
            # we need to specify --config Release at build time
            if sys.platform.startswith("win"):
                # Specify platform x86 or x86-64
                platform = "x64" if sys.maxsize > 2 ** 32 else "Win32"
                cmake_config_args += ["-A", platform]
                cmake_build_args += ["--config", cfg]

            cmake_config_args += shlex.split(os.environ.get("ALE_PY_CMAKE_ARGS", ""))
            cmake_build_args += shlex.split(os.environ.get("ALE_PY_BUILD_ARGS", ""))

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            subprocess.check_call(
                ["cmake", ext.sourcedir] + cmake_config_args, cwd=self.build_temp
            )
            subprocess.check_call(
                ["cmake", "--build", "."] + cmake_build_args, cwd=self.build_temp
            )


def _read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()


def _is_valid_semver(version):
    """
    Checks if `version` conforms to semver rules.
    """
    regex = r"^((([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?)(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?)$"
    return re.match(regex, version)


def _parse_version(filename):
    """
    Parse ALEVERSION from `CMakeLists.txt`

    args:
        filename: should point to the projects CMakeLists.txt
    returns:
        version:
            1) Running locally version will be of the form VERSION.dev
            2) Running in CI with a version tag will be of the form TAGGED_VERSION
    raises:
        RuntimeError:
            1) Unable to find ALEVERSION in `filename` 
        AssertionError:
            1) Running in CI and tagged version doesn't match parsed version
            2) Tagged version or parsed version doesn't conform to semver rules
    """
    # Parse version from file
    contents = _read(filename)
    version_match = re.search(r"ALEVERSION\s\"(\d+.*)\"", contents, re.M)
    if not version_match:
        raise RuntimeError("Unable to find ALEVERSION in %s" % filename)

    version = version_match.group(1)
    version_suffix = ".dev"
    assert _is_valid_semver(version), "ALEVERSION %s must conform to semver." % version

    # If the git ref is a tag verify the tag and don't use a suffix
    ref = "GITHUB_REF"
    tag_regex = r"refs\/tags\/v(.*)$"
    if os.environ.get(ref, False) and re.match(tag_regex, os.environ.get(ref)):
        version_match = re.search(tag_regex, os.environ.get(ref))
        version_tag = version_match.group(1)
        assert _is_valid_semver(version_tag), (
            "Tag is invalid semver. %s must conform to semver." % version_tag
        )
        assert version_tag == version, (
            "Tagged version must match ALEVERSION but got:\n\tALEVERSION: %s\n\tTAG: %s"
            % (version, version_tag)
        )
        version_suffix = ""

    return version + version_suffix


setup(
    name="ale-py",
    version=_parse_version("CMakeLists.txt"),
    description="Arcade Learning Environment Python Interface",
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    keywords=["reinforcement-learning", "arcade-learning-environment", "atari"],
    url="https://github.com/mgbellemare/Arcade-Learning-Environment",
    author="Arcade Learning Environment Authors",
    license="GPL",
    ext_modules=[
        CMakeExtension(
            "ale_py.libale_c",
            ".",
            [
                "-DUSE_SDL=OFF",
                "-DUSE_RLGLUE=OFF",
                "-DBUILD_EXAMPLES=OFF",
                "-DBUILD_CPP_LIB=OFF",
                "-DBUILD_CLI=OFF",
                "-DBUILD_C_LIB=ON",
            ],
        )
    ],
    cmdclass={"build_ext": CMakeBuild},
    packages=["ale_py"],
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
