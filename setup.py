"""Setup file for ALE."""

import os
import platform
import re
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

current_working_file = os.path.abspath(os.path.dirname(__file__))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    PLAT_TO_CMAKE = {
        "win32": "Win32",
        "win-amd64": "x64",
        "win-arm32": "ARM",
        "win-arm64": "ARM64",
    }

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for rpath detection of libraries
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        config = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DSDL_SUPPORT=ON",
            "-DSDL_DYNLOAD=ON",
            "-DBUILD_CPP_LIB=OFF",
            "-DBUILD_PYTHON_LIB=ON",
            "-DBUILD_VECTOR_LIB=ON",
        ]
        # We can only support XLA on platforms with JAX support (e.g., 3.10+ and Windows/Linux)
        if sys.version_info >= (3, 10) and platform.system() in {
            "Windows",
            "Linux",
        }:
            cmake_args.append("-DBUILD_VECTOR_XLA_LIB=ON")
        build_args = []

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja  # noqa: F401

                    ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    if shutil.which("ninja") is not None:
                        cmake_args += [
                            "-GNinja",
                        ]

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", self.PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={extdir}"
                ]
                build_args += ["--config", config]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


def parse_version(version_file):
    """Parse version from `version_file`.

    If we're running on CI, i.e., CIBUILDWHEEL is set, then we'll parse
    the version from `GITHUB_REF` using the official semver regex.

    If we're not running in CI we'll append the current git SHA to the
    version identifier.

    raises AssertionError: If `${GITHUB_REF#/v/*/}` doesn't start with
        the version specified in `version_file`.
    """
    semver_regex = r"(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?"
    semver_prog = re.compile(semver_regex)

    with open(version_file) as fp:
        version = fp.read().strip()
        assert semver_prog.match(version) is not None

    return version


def get_setup_requires():
    """Determine extra dependencies necessary for build"""
    setup_requires = []
    if shutil.which("cmake") is None:
        setup_requires += ["cmake>=3.22"]
    if shutil.which("ninja") is None:
        setup_requires += [
            "ninja; sys_platform != 'win32' and platform_machine != 'arm64'"
        ]
    return setup_requires


if __name__ == "__main__":
    # Allow for running `pip wheel` from other directories
    current_working_file and os.chdir(current_working_file)
    # Most config options are in `setup.cfg`. These are the
    # only dynamic options we need at build time.
    setup(
        name="ale-py",
        version=parse_version("version.txt"),
        setup_requires=get_setup_requires(),
        ext_modules=[CMakeExtension("ale_py._ale_py")],
        cmdclass={"build_ext": CMakeBuild},
    )
