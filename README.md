The Arcade Learning Environment
<img align="right" src="docs/manual/figures/ale.svg" width=75>
===============================

![Build Status](https://github.com/mgbellemare/Arcade-Learning-Environment/workflows/Build%20ALE/badge.svg)

**The Arcade Learning Environment (ALE) is a simple object-oriented framework that allows researchers and hobbyists to develop AI agents for Atari 2600 games.**
It is built on top of the Atari 2600 emulator [Stella](https://stella-emu.github.io) and separates the details of emulation from agent design.
This [video](https://www.youtube.com/watch?v=nzUiEkasXZI) depicts over 50 games currently supported in the ALE.

For an overview of our goals for the ALE read [The Arcade Learning Environment: An Evaluation Platform for General Agents](https://jair.org/index.php/jair/article/view/10819).
If you use ALE in your research, we ask that you please cite this paper in reference to the environment. See the [Citing](#Citing) section for BibTeX entries.

Features
--------

- Object-oriented framework with support to add agents and games.
- Emulation core uncoupled from rendering and sound generation modules for fast
  emulation with minimal library dependencies.
- Automatic extraction of game score and end-of-game signal for more than 50
  Atari 2600 games.
- Multi-platform code (compiled and tested under macOS, Windows, and several Linux
  distributions).
- Python development is supported through [pybind11](https://github.com/pybind/pybind11).
- Agents programmed in C++ have access to all features in the ALE.
- Visualization tools.

Quick Start
===========

You must have a valid C++17 compiler and the following dependencies installed (we recommend using [`vcpkg`](https://github.com/microsoft/vcpkg) on all platforms)

```sh
vcpkg install zlib sdl1
```

Note: `sdl` is optional but can be useful for display/audio support (i.e., `display_screen` and `sound` config options).

Python
------

The package `ale-py` will be distributed via PyPi but for the time being Python users can install the ALE via

```sh
pip install .
```
Note: Make sure you're using an up to date version of `pip` or the install may fail.


You can now import the ALE in your Python projects with
```python
from ale_py import ALEInterface

ale = ALEInterface()
ale.loadROM(...)
```

C++
---

We use CMake as a first class citizen and you can use the ALE directly with any CMake project.
To compile and install the ALE you can run

```sh
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install
```

There are optional flags `-DUSE_SDL=ON/OFF` to toggle SDL support (`OFF` by default), `-DBUILD_CPP_LIB=ON/OFF` to build
the `ale-lib` C++ target (`ON` by default), and `-DBUILD_PYTHON_LIB=ON/OFF` to build the pybind11 wrapper (`ON` by default).

Finally you can link agaisnt the ALE in your own CMake project as follows

```cmake
find_package(ale REQUIRED)
target_link_libraries(YourTarget ale::ale-lib)
```

Citing
======

If you use the ALE in your research, we ask that you please cite the following.

*M. G. Bellemare, Y. Naddaf, J. Veness and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013.*

In BibTeX format:

```bibtex
@Article{bellemare13arcade,
    author = {{Bellemare}, M.~G. and {Naddaf}, Y. and {Veness}, J. and {Bowling}, M.},
    title = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
    journal = {Journal of Artificial Intelligence Research},
    year = "2013",
    month = "jun",
    volume = "47",
    pages = "253--279",
}
```

If you use the ALE with sticky actions (flag ``repeat_action_probability``), or if
you use the different game flavours (mode and difficulty switches), we ask you
that you also cite the following:

*M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. J. Hausknecht, M. Bowling. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents,  Journal of Artificial Intelligence Research, Volume 61, pages 523-562, 2018.*

In BibTex format:

```bibtex
@Article{machado18arcade,
    author = {Marlos C. Machado and Marc G. Bellemare and Erik Talvitie and Joel Veness and Matthew J. Hausknecht and Michael Bowling},
    title = {Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents},
    journal = {Journal of Artificial Intelligence Research},
    volume = {61},
    pages = {523--562},
    year = {2018}
}
```
