

# The Multi-Agent Arcade Learning Environment


## Overview

This is a fork of the [Arcade Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment). It is mostly backwards compatible with ALE and it also supports certain games with 2 and 4 players.

If you use ALE in your
research, we ask that you please cite this paper in reference to the environment
(BibTeX entry at the end of this document). Also, if you have any questions or
comments about the ALE, please contact us through our [mailing
list](https://groups.google.com/forum/#!forum/arcade-learning-environment).

Feedback and suggestions are welcome and may be addressed to any active member
of the ALE team.

## Features

- Object-oriented framework with support to add agents and games.
- Emulation core uncoupled from rendering and sound generation modules for fast
  emulation with minimal library dependencies.
- Automatic extraction of game score and end-of-game signal for more than 50
  Atari 2600 games.
- Multi-platform code (compiled and tested under OS X and several Linux
  distributions, with Cygwin support).
- Python development is supported through ctypes.
- Agents programmed in C++ have access to all features in the ALE.
- Visualization tools.

## Quick start

Install main dependences:

```
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
```

Compilation:

```
$ mkdir build && cd build
$ cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
$ make -j 4
```

To install the Python module:

```
$ pip install multi-agent-ale-py
```



Getting the ALE to work on Visual Studio requires a bit of extra wrangling. You
may wish to use IslandMan93's [Visual Studio port of the
ALE.](https://github.com/Islandman93/Arcade-Learning-Environment)

For more details and installation instructions, see the [manual](doc/manual/manual.pdf).
To ask questions and discuss, please join the
[ALE-users group](https://groups.google.com/forum/#!forum/arcade-learning-environment).


## Citing The Arcade Learning Environment

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

If you use the ALE with sticky actions (flag `repeat_action_probability`), or if
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

## Contributing, code style

If you would like to make changes to the codebase, please adhere to the
following code style conventions.

ALE contains two sets of source files: Files .hxx and .cxx are part of the
Stella emulator code. Files .hpp and .cpp are original ALE code. The Stella
files are not subject to our conventions, please retain their local style.

The ALE code style conventions are roughly summarised as "clang-format with the
following settings: ReflowComments: false, PointerAlignment: Left,
KeepEmptyLinesAtTheStartOfBlocks: false, IndentCaseLabels: true,
AccessModifierOffset: -1". That is:

- Indent by two spaces; Egyptian braces, no extraneous newlines at the margins
  of blocks and between top-level declarations.
- Pointer/ref qualifiers go on the left (e.g. `void* p`).
- Class member access modifiers are indented by _one_ space.
- Inline comments should be separated from code by two spaces (though this is
  not currently applied consistently).
- There is no strict line length limit, but keep it reasonable.
- Namespace close braces and `#endif`s should have comments.

The overall format should look reasonably "compact" without being crowded. Use
blank lines generously _within_ blocks and long comments to create visual cues
for the segmentation of ideas.
