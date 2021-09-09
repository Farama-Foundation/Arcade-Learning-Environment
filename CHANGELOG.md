# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Native support for OpenAI Gym
- Native Python interface using pybind11 which results in a speedup for Python workloads as well as proper support for objects like `ALEState`
- Python ROM management, e.g., `ale-import-roms`
- PyPi Python wheels published as `ale-py` + we distribute SDL2 for out of the box visualization + audio support
- `isSupportedROM(path)` to check if a ROM file is supported by the ALE
- Added new games: Atlantis2, Backgammon, BasicMath, Blackjack, Casino, Crossbow, DarkChambers, Earthworld, Entombed, ET, FlagCapture, Hangman, HauntedHouse, HumanCannonball, Klax, MarioBros, MiniatureGolf, Othello, Pacman, Pitfall2, SpaceWar, Superman, Surround, TicTacToe3D, VideoCheckers, VideoChess, VideoCube, WordZapper (thanks @tkoppe)
- Added (additional) mode/difficulty settings for: Lost Luggage, Turmoil, Tron Dead Discs, Pong, Mr. Do, King Kong, Frogger, Adventure (thanks @tkoppe)
- Added `cloneState(include_rng)` which will eventually replace `cloneSystemState` (behind the scenes `cloneSystemState` is equivalent to `cloneState(include_rng=True)`).

### Changed
- Rewrote SDL support using SDL2 primitives
- SDL2 now renders every frame independent of frameskip
- SDL2 renders at the proper ROM framerate (added benefit of audio sync support)
- Rewrote entire CMake infrastructure which now supports vcpkg natively
- C++ minimum version is now C++17
- Changed all relative imports to absolute imports
- Switched from Travis CI to Github Actions
- Allow for paddle controller's min/max setting to be configurable
- More robust version handling between C++ & Python distributions
- Updated Markdown documentation to replace TeX manual

### Fixed
- Fixed bankswitching type for UA cartridges
- Fixed a SwapPort bug in Surround
- Fixed multiple bugs in handling invalid ROM files (thanks @tkoeppe)
- Fixed initialization of TIA static data to make it thread safe (thanks @tkoeppe)
- Fixed RNG initialization, this was one of the last barriers to making the ALE fully deterministic, we are now fully deterministic

### Removed
- Removed FIFO interface
- Removed RL-GLUE support
- Removed ALE CLI interface
- Removed Java interface
- Removed `ALEInterface::load()`, `ALEInterface::save()`. If you require this stack functionality it's easy to implement on your own using `ALEInterface::cloneState(include_rng)`
- Removed os-dependent filesystem code in favour of C++17 `std::fs`
- Removed human control mode
- Removed old makefile build system in favour of CMake
- Removed bspf
- Removed unused controller types: Driving, Booster, Keyboard
- Removed AtariVox
- Removed Stella types (e.g., Array) in favour of STL types
- Remove Stella debugger
- Remove Stella CheatManager
- Lots of code cleanups conforming to best practices (thanks @tkoeppe)


## [0.6.1] - 2019-11-20
### Changed
- Speedup of up to 30% by optimizing variable types (@qstanczyk)

### Fixed
- Fixed switch fall-through with Gravitar lives detection (@lespeholt)

## [0.6.0] - 2015-06-23
### Added
- Support for modes and difficulties in Atari games (@mcmachado)
- Frame maxpooling as a post-processing option (@skylian)
- Added support for: Turmoil, Koolaid, Tron Deadly Discs, Mr. Do, Donkey Kong, Keystone Kapers, Frogger, Sir Lancelot, Laser Gates, Lost Luggage,
- Added MD5 list of supported ROMs

### Changed
- Disabled color averaging by default
- Replaced TinyMT with C++11 random

### Fixed
- Fixed old color averaging scheme (PR #181)
- Fixed minimal action set in Pong
- Fixed termination issues in Q*Bert


## [0.5.2] - 2015-10-04
### Added
- Routines for ALEState serialization (@Jragonmiris).

### Changed
- Enforce flags existence (@mcmachado).

### Fixed
- Fix RNG issues introduced in 0.5.0.
- Additional bug fixes.


## [0.5.1] - 2015-07-07
### Added
- Added RNG serialization capability.

### Changed
- Refactored Python getScreenRGB to return unpacked RGB values (@spragunr).
- Sets the default value of the color_averaging flag to be true. It was true by default in previous versions but was changed in 0.5.0. Reverted for backward compatibility.

### Fixed
- Bug fixes from ALE 0.5.0.


## [0.5.0] - 2015-06-22
### Added
- Added action_repeat_stochasticity.
- Added sound playback, visualization.
- Added screen/sound recording ability.
- CMake now available.
- Incorporated Benjamin Goodrich's Python interface.
- Added examples for shared library, Python, fifo, RL-Glue interfaces.
- Incorporated Java agent into main repository.

### Changed
- Better ALEInterface.
- Many other changes.

### Fixed
- Some game fixes.

### Removed
- Removed internal controller, now superseded by shared library interface.
- Removed the following command-line flags: 'output_file', 'system_reset_steps', 'use_environment_distribution', 'backward_compatible_save', internal agent flags
- The flag 'use_starting_actions' was removed and internally its value is always 'true'.
- The flag 'disable_color_averaging' was renamed to 'color_averaging' and FALSE is its default value.


## [0.4.4] - 2014-04-28
### Fixed
- Fixed a memory issue in ALEScreen.


## [0.4.3] - 2014-04-26
### Fixed
- Fixed issues with frame numbers not being correctly updated.
- Fixed a bug where total reward was not properly reported under frame skipping. 
- Fixed a bug with ALEState's m_frame_number.


## [0.4.2] - 2013-06-12
### Changed
- Modified StellaEnvironment save/load interface to provide additional flexibility. 

### Fixed
- Series of bug fixes from Matthew Hausknecht and community.


## [0.4.1] - 2013-05-24
### Added
- Added frame skipping support directly in StellaEnvironment.

### Changed
- Reverted default number of episodes to 10.

### Fixed
- Fixed RL-Glue syntax from OBSERVATION to OBSERVATIONS. Thanks to Angus MacIsaac for picking this bug up.


## [0.4.0] - 2013-04-22
### Added
- RL-Glue support
- Shared library interface
- Simpler direct environment interfacing
- Improved environment handling
- Improved environment customization
- Better documentation


## 0.3.0 - 2012-07-22
- Initial ALE release.


[unreleased]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/aa433a4b401bc3e7113c494edfc90500bc4afc78...v0.5.0
[0.4.4]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/d93189e0f00b5cb10120134ca965d8a5d3124581...aa433a4b401bc3e7113c494edfc90500bc4afc78
[0.4.3]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/b905e07ead43d07f386b35128e7eff60595e1581...d93189e0f00b5cb10120134ca965d8a5d3124581
[0.4.2]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/ba33f16376b545462666268194e8f72df82c1a3a...b905e07ead43d07f386b35128e7eff60595e1581
[0.4.1]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/84f9678d713695314570e0f183072f36e177a364...ba33f16376b545462666268194e8f72df82c1a3a
[0.4.0]: https://github.com/mgbellemare/Arcade-Learning-Environment/compare/5c45f643a78ef96ade23928fd6a3740172ec1e35...84f9678d713695314570e0f183072f36e177a364
