import distutils.dir_util
import os
import sys
import platform
import multiprocessing
import subprocess
import shlex
import re

from distutils.command.clean import clean as _clean
from setuptools import setup, Extension, Distribution as _distribution
from setuptools.command.build_ext import _build_ext
from shutil import rmtree

system = platform.system()
here = os.path.abspath(os.path.dirname(__file__))


class Distribution(_distribution):
    global_options = _distribution.global_options

    global_options += [("cmake-options=", None, "Additional semicolon-separated cmake setup options list")]

    if system == "Windows":
        global_options += [("vcpkg-root=", None, "Path to vcpkg root. For Windows only")]

    def __init__(self, attrs=None):
        self.vcpkg_root = None
        self.cmake_options = None
        super().__init__(attrs)


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class BuildALEPythonInterface(_build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

        super().run()

    def build_extension(self, ext):
        distutils.dir_util.mkpath(self.build_temp)

        libdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        distutils.dir_util.mkpath(libdir)

        config = "Debug" if self.debug else "Release"

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(config),
            "-DUSE_SDL=OFF",
            "-DBUILD_CPP_LIB=OFF",
            "-DBUILD_PYTHON=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(config.upper(), libdir),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(config.upper(), libdir),
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(config.upper(), libdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
        ]
        build_args = [
            "--config",
            config,
            "--parallel",
            str(multiprocessing.cpu_count()),
        ]

        if self.distribution.cmake_options is not None:
            cmake_args += shlex.split(self.distribution.cmake_args)
        cmake_args += shlex.split(os.environ.get("ALE_PY_CMAKE_ARGS", ""))

        if system == "Windows":
            platform = "x64" if sys.maxsize > 2 ** 32 else "Win32"
            cmake_args += ["-A", platform]

            if self.distribution.vcpkg_root is not None:
                abs_vcpkg_path = os.path.abspath(self.distribution.vcpkg_root)
                vcpkg_toolchain = os.path.join(
                    abs_vcpkg_path, "scripts", "buildsystems", "vcpkg.cmake"
                )
                cmake_args += ["-DCMAKE_TOOLCHAIN_FILE=" + vcpkg_toolchain]

        os.chdir(self.build_temp)
        self.spawn(["cmake", here] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(here)


class Clean(_clean):
    """Hook to clean up after building the Python package."""

    def run(self):
        rmtree(os.path.join(here, "dist"), ignore_errors=True)
        rmtree(os.path.join(here, "build"), ignore_errors=True)
        rmtree(os.path.join(here, "ale_py.egg-info"), ignore_errors=True)
        super().run()


def _read(filename):
    """Reads an entire file into a string."""
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()


def _is_valid_semver(version):
    """Checks if `version` conforms to semver rules."""
    regex = r"^((([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?)(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?)$"
    return re.match(regex, version)


def parse_version(version_file):
    """
    Parse version from `version_file`.

    If `ALE_BUILD_VERSION` is specified this version overrides that in
    `version_file`.

    If `ALE_BUILD_VERSION` is not specified the git sha will be appended
    to the version identifier.

    raises AssertionError: If `ALE_BUILD_VERSION` doesn't start with the version
        specified in `version_file`
    """
    version = open(version_file).read().strip()

    if os.getenv('ALE_BUILD_VERSION'):
        assert os.getenv('ALE_BUILD_VERSION').startswith(version)
        version = os.getenv('ALE_BUILD_VERSION')
    else:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=here).decode('ascii').strip()
        version += f"+{sha}"

    return version


setup(
    name="ale-py",
    version=parse_version("version.txt"),
    description="The Arcade Learning Environment (ALE) - a platform for AI research.",
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    keywords=["reinforcement-learning", "arcade-learning-environment", "atari"],
    url="https://github.com/mgbellemare/Arcade-Learning-Environment",
    author="Arcade Learning Environment Authors",
    license="GPLv2",
    python_requires=">=3.5",
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        # Audience
        "Intended Audience :: Science/Research",
        # License
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        # Language Support
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        # Topics
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
    include_package_data=True,
    distclass=Distribution,
    ext_modules=[CMakeExtension("ale-py")],
    cmdclass={"build_ext": BuildALEPythonInterface, "clean": Clean},
    install_requires=["numpy"],
    test_requires=["pytest"],
)
