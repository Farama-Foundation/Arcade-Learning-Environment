The Arcade Learning Environment
<a href="#the-arcade-learning-environment">
  <img alt="Arcade Learning Environment" align="right" src="docs/static/ale.svg" width=75 />
</a>
===============================

[![Continuous Integration](https://github.com/mgbellemare/Arcade-Learning-Environment/actions/workflows/ci.yml/badge.svg)](https://github.com/mgbellemare/Arcade-Learning-Environment/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/ale-py)](https://pypi.org/project/ale-py)


**The Arcade Learning Environment (ALE) is a simple framework that allows researchers and hobbyists to develop AI agents for Atari 2600 games.**
It is built on top of the Atari 2600 emulator [Stella](https://stella-emu.github.io) and separates the details of emulation from agent design.
This [video](https://www.youtube.com/watch?v=nzUiEkasXZI) depicts over 50 games currently supported in the ALE.

For an overview of our goals for the ALE read [The Arcade Learning Environment: An Evaluation Platform for General Agents](https://jair.org/index.php/jair/article/view/10819).
If you use ALE in your research, we ask that you please cite this paper in reference to the environment. See the [Citing](#Citing) section for BibTeX entries.

Features
--------

- Object-oriented framework with support to add agents and games.
- Emulation core uncoupled from rendering and sound generation modules for fast
  emulation with minimal library dependencies.
- Automatic extraction of game score and end-of-game signal for more than 100
  Atari 2600 games.
- Multi-platform code (compiled and tested under macOS, Windows, and several Linux distributions).
- Python bindings through [pybind11](https://github.com/pybind/pybind11).
- Native support for OpenAI Gym.
- Visualization tools.

Quick Start
===========

The ALE currently supports three different interfaces: C++, Python, and OpenAI Gym.


Python
------

You simply need to install the `ale-py` package distributed via PyPI:

```shell
pip install ale-py
```
Note: Make sure you're using an up to date version of `pip` or the install may fail.


You can now import the ALE in your Python projects with
```python
from ale_py import ALEInterface

ale = ALEInterface()
```

### ROM Management

The ALE doesn't distribute ROMs but we do provide a couple tools for managing your ROMs. First is the command line tool `ale-import-roms`. You can simply specify a directory as the first argument to this tool and we'll import all supported ROMs by the ALE.

```shell
ale-import-roms roms/

[SUPPORTED]       breakout   roms/breakout.bin
[SUPPORTED]       freeway    roms/freeway.bin

[NOT SUPPORTED]              roms/custom.bin

Imported 2/3 ROMs
```
Furthermore, Python packages can expose ROMs for discovery using the special `ale-py.roms` entry point. For more details check out the example [python-rom-package](./examples/python-rom-package).

Once you've imported a supported ROM you can simply import the path from the `ale-py.roms` package and load the ROM in the ALE:
```py
from ale_py.roms import Breakout

ale.loadROM(Breakout)
```

## OpenAI Gym

Gym support is included in `ale-py`. Simply install  the Python package using the instructions above. You can also install `gym[atari]` which also installs `ale-py` with Gym.

As of Gym v0.20 and onwards all Atari environments are provided via `ale-py`. We do recommend using the new `v5` environments in the `ALE` namespace:

```py
import gym

env = gym.make('ALE/Breakout-v5')
```
The `v5` environments follow the latest methodology set out in [Revisiting the Arcade Learning Environment by Machado et al.](https://jair.org/index.php/jair/article/view/11182).

The only major change difference from Gym's `AtariEnv` is that we'd recommend not using the `env.render()` method in favour of supplying the `render_mode` keyword argument during environment initialization. The `human` render mode will give you the advantage of: frame perfect rendering, audio support, and proper resolution scaling. For more information check out [docs/gym-interface.md](./docs/gym-interface.md).

For more information on changes to the Atari environments in OpenAI Gym please check out [the following blog post](https://brosa.ca/blog/ale-release-v0.7).

C++
---

The following instructions will assume you have a valid C++17 compiler and [`vcpkg`](https://github.com/microsoft/vcpkg) installed.

We use CMake as a first class citizen, and you can use the ALE directly with any CMake project.
To compile and install the ALE you can run

```sh
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install
```

There are optional flags `-DSDL_SUPPORT=ON/OFF` to toggle SDL support (i.e., `display_screen` and `sound` support; `OFF` by default), `-DBUILD_CPP_LIB=ON/OFF` to build
the `ale-lib` C++ target (`ON` by default), and `-DBUILD_PYTHON_LIB=ON/OFF` to build the pybind11 wrapper (`ON` by default).

Finally, you can link agaisnt the ALE in your own CMake project as follows

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
