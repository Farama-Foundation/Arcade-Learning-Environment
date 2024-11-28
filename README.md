The Arcade Learning Environment
<a href="#the-arcade-learning-environment">
  <img alt="Arcade Learning Environment" align="right" width=75 src="https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/docs/_static/img/ale.svg" />
</a>
===============================

[![Python](https://img.shields.io/pypi/pyversions/ale-py.svg)](https://badge.fury.io/py/ale-py)
[![PyPI Version](https://img.shields.io/pypi/v/ale-py)](https://pypi.org/project/ale-py)

**The Arcade Learning Environment (ALE) is a simple framework that allows researchers and hobbyists to develop AI agents for Atari 2600 games.**
It is built on top of the Atari 2600 emulator [Stella](https://stella-emu.github.io) and separates the details of emulation from agent design.
This [video](https://www.youtube.com/watch?v=nzUiEkasXZI) depicts over 50 games currently supported in the ALE.

For an overview of our goals for the ALE read [The Arcade Learning Environment: An Evaluation Platform for General Agents](https://jair.org/index.php/jair/article/view/10819).
If you use ALE in your research, we ask that you please cite this paper in reference to the environment. See the [Citing](#Citing) section for BibTeX entries.

Features
--------

- Object-oriented framework with support to add agents and games.
- Emulation core uncoupled from rendering and sound generation modules for fast emulation with minimal library dependencies.
- Automatic extraction of game score and end-of-game signal for more than 100  Atari 2600 games.
- Multi-platform code (compiled and tested under macOS, Windows, and several Linux distributions).
- Python bindings through [pybind11](https://github.com/pybind/pybind11).
- Native support for [Gymnasium](http://github.com/farama-Foundation/gymnasium), a maintained fork of OpenAI Gym.
- Visualization tools.
- Atari roms are packaged within the pip package

Quick Start
===========

The ALE currently supports three different interfaces: C++, Python, and Gymnasium.

Python
------

You simply need to install the `ale-py` package distributed via PyPI:

```shell
pip install ale-py
```
Note: Make sure you're using an up-to-date version of `pip` or the installation may fail.

You can now import the ALE in your Python projects with providing a direct interface to Stella for interacting with games
```python
from ale_py import ALEInterface, roms

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("breakout"))
ale.reset_game()

reward = ale.act(0)  # noop
screen_obs = ale.getScreenRGB()
```

## Gymnasium

For simplicity for installing ale-py with Gymnasium, `pip install "gymnasium[atari]"` shall install all necessary modules and ROMs. See Gymnasium [introductory page](https://gymnasium.farama.org/main/introduction/basic_usage/) for description of the API to interface with the environment.

```py
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

env = gym.make('ALE/Breakout-v5', render_mode="human")  # remove render_mode in training
obs, info = env.reset()
episode_over = False
while not episode_over:
    action = policy(obs)  # to implement - use `env.action_space.sample()` for a random policy
    obs, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated
env.close()
```

To run with continuous actions, you can simply modify the call to `gym.make` above with:
```python
env = gym.make('ALE/Breakout-v5', continuous=True, render_mode="human")
```

For all the environments available and their description, see [gymnasium atari page](https://gymnasium.farama.org/environments/atari/).

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

Finally, you can link against the ALE in your own CMake project as follows

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

If you use the CALE (Continuous ALE), we ask you that you also cite the following:

*Jesse Farebrother and Pablo Samuel Castro.  Cale:  Continuous arcade learning environment.Ad-vances in Neural Information Processing Systems, 2024.*

In BibTex format:

```bibtex
@article{farebrother2024cale,
  title={C{ALE}: Continuous Arcade Learning Environment},
  author={Jesse Farebrother and Pablo Samuel Castro},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
