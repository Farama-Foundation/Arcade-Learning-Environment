# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.10.2...v0.11.0) - 2025-04-26

This release adds (an experiment) built-in vectorisation environment, available through `gymnasium.make_vec("ALE/{game_name}-v5", num_envs)` or `ale_py.AtariVectorEnv("{rom_name}", num_envs)`.

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

envs = gym.make_vec("ALE/Pong-v5")
observations, infos = envs.reset()

for i in range(100):
  actions = envs.action_space.sample()
  observations, rewards, terminations, truncations, infos = envs.step(actions)

envs.close()
```

Vectorisation is a crucial feature of RL to help increase the sample rate of environments through sampling multiple sub-environments at the same time.
[Gymnasium](https://gymnasium.farama.org/api/vector/) provides a generalised vectorisation capability, however, is relatively slow due its python implementation.
For faster implementations, [EnvPool](https://github.com/sail-sg/envpool) provide C++ vectorisation that significantly increase the sample speed but it no longer maintained.
Inspired by the `EnvPool` implementation, we've implemented an asynchronous vectorisation environment in C++, in particular, the [standard Atari preprocessing](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing) including frame skipping, frame stacking, observation resizing, etc.

For full documentation of the vector environment, see [this page](https://ale.farama.org/v0.11.0/vector-environment).

We will continue building out this vectorisation to include [XLA](https://github.com/openxla/xla) support, improved preprocessing and auto resetting.

As this is an experimental feature, we wish to hear about any bugs, problems or features to add. Raise an issue on GitHub or ask a question on the [Farama Discord server](https://discord.gg/bnJ6kubTg6).

## [0.10.2](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.10.1...v0.10.2) - 2025-02-13

Fixed performance regression for CPP users - A single-argument `act` function was missing causing the `paddle_strength` introduced in v0.10.0 to default to zero rather than one. As Gymnasium passed this variable to act, this was only an issue for users directly interacting with `ale_interface`.  For more details, see https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/595.

## [0.10.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.10.0...v0.10.1) - 2024-10-26

Revert change to requirements that `numpy < 2.0` and add support for building from source distribution, `tar.gz` (though not recommended).

## [0.10.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.9.1...v0.10.0) - 2024-10-23

In v0.10, ALE now has its own dedicated website, https://ale.farama.org/ with Atari's documentation being moved from Gymnasium.

### Continuous ALE
Previously in the original ALE interface, the actions are only joystick ActionEnum inputs.
Then, for games that use a paddle instead of a joystick, joystick controls are mapped into discrete actions applied to paddles, ie:
- All left actions (`LEFTDOWN`, `LEFTUP`, `LEFT...`) -> paddle left max
- All right actions (`RIGHTDOWN`, `RIGHTUP`, `RIGHT...`) -> paddle right max
- Up... etc.
- Down... etc.

This results in loss of continuous action for paddles.
This change keeps this functionality and interface, but allows for continuous action inputs for games that allow paddle usage.

To do that, the CPP interface has been modified.

_Old Discrete ALE interface_
```cpp
reward_t ALEInterface::act(Action action)
```

_New Mixed Discrete-Continuous ALE interface_
```cpp
reward_t ALEInterface::act(Action action, float paddle_strength = 1.0)
```

Games where the paddle is not used simply have the `paddle_strength` parameter ignored.
This mirrors the real world scenario where you have a paddle connected, but the game doesn't react to it when the paddle is turned.
This maintains backwards compatibility.

The Python interface has also been updated.

_Old Discrete ALE Python Interface_
```py
ale.act(action: int)
```

_New Mixed Discrete-Continuous ALE Python Interface_
```py
ale.act(action: int, strength: float = 1.0)
```

More specifically, when continuous action space is used within an ALE gymnasium environment, discretization happens at the Python level.
```py
if continuous:
    # action is expected to be a [2,] array of floats
    x, y = action[0] * np.cos(action[1]), action[0] * np.sin(action[1])
    action_idx = self.map_action_idx(
        left_center_right=(
            -int(x < self.continuous_action_threshold)
            + int(x > self.continuous_action_threshold)
        ),
        down_center_up=(
            -int(y < self.continuous_action_threshold)
            + int(y > self.continuous_action_threshold)
        ),
        fire=(action[-1] > self.continuous_action_threshold),
    )
    ale.act(action_idx, action[1])
```

More specifically, [`self.map_action_idx`](https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/550/files#diff-057906329e72d689f1d4d9d9e3f80df11ffe74da581b29b3838a436e90841b5cR388-R447) is an `lru_cache`-ed function that takes the continuous action direction and maps it into an ActionEnum.

### Other changes

We have moved the project main code from `src` into `src/ale` to help incorporate ALE into c++ project and in the python API, we have updated `get_keys_to_action` to work with `gymnasium.utils.play` through changing the key for no-op from `None` to the `e` key.

## [0.9.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.9.0...v0.9.1) - 2024-09-01

Added support for Numpy 2.0.

## [0.9.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.8.1...v0.9.0) - 2024-05-10

Previously, ALE implemented only a [Gym](https://github.com/openai/gym) based environment, however, as Gym is no longer maintained (last commit was 18 months ago). We have updated `ale-py` to use [Gymnasium](http://github.com/farama-Foundation/gymnasium) (a maintained fork of Gym) as the sole backend environment implementation. For more information on Gymnasium’s API, see their [introduction page](https://gymnasium.farama.org/main/introduction/basic_usage/).

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)  # this is unnecessary but prevents IDE complaining

env = gym.make("ALE/Pong-v5", render_mode="human")

obs, info = env.reset()
episode_over = False
while not episode_over:
	action = policy(obs)  # replace with actual policy
	obs, reward, terminated, truncated, info = env.step(action)
	episode_over = terminated or truncated
env.close()
```

An important change in this update is that the Atari ROMs are packaged within the PyPI installation such that users no longer require AutoROM or `ale-import-roms` for downloading or loading ROMs. This should significantly simplify installing Atari for users. For users that wish to load ROMs from an alternative folder, use the `ALE_ROM_DIR` system environment variable to specify a folder directory.

Importantly, Gymnasium 1.0.0 removes a registration plugin system that ale-py utilises where atari environments would be registered behind the scene. As a result, projects will need to import `ale_py`, to register all the atari environments, before an atari environment can be created with `gymnasium.make`. For example, see below

### Other changes
- Added Python 3.12 support.
- Replace interactive exit by sys.exit (https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/498)
- Fix C++ documentation example links(https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/501)
- Add support for gcc 13 (https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/503)
- Unpin cmake dependency and remove wheel from build system (https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/493)
- Add missing imports for cstdint (https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/486)
- Allow installing without git (https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/492)
- Update to require `importlib-resources` for < 3.9  (https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/491)

## [0.8.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.8.0...v0.8.1) - 2023-02-17

### Added

- Added type stubs for the native ALE Python module generated via pybind11. You'll now get type hints in your IDE.

### Fixed

- Fixed `render_mode` attribute on legacy Gym environment (@younik)
- Fixed a bug which could parse invalid ROM names containing numbers, e.g., TicTacToe3D or Pitfall2
- Changed the ROM identifier of VideoChess & VideoCube to match VideoCheckers & VideoPinball.
  Specifically, the environment ID changed from `Videochess` -> `VideoChess` and `Videocube` -> `VideoCube`.
  Most ROMs had the ID correctly as `video_chess.bin` and `video_cube.bin` but for those who didn't you can
  simply run `ale-import-roms` which will automatically correct this for you.
- Reverted back to manylinux2014 (glibc 2.17) to better support older operating systems.

## [0.8.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.7.5...v0.8.0) - 2022-09-05

### Added

- Added compliance with the Gym v26 API. This includes multiple breaking changes to the Gym API. See the [Gym Release](https://github.com/openai/gym) for additional information.
- Reworked the ROM plugin API resulting in reduced startup time when importing `ale_py.roms`.
- Added a truncation API to the ALE interface to query whether an episode was truncated or terminated (`ale.game_over(with_truncation=true/false)` and `ale.game_truncated()`)
- Added proper Gym truncation on max episode frames. This no longer relies on the `TimeLimit` wrapper with the new truncation API in Gym v26.
- Added a setting for truncating on loss-of-life.
- Added a setting for clamping rewards.
- Added `const` keywords to attributes in `ale::ALEInterface` (#457) (@AlessioZanga).
- Added explicit exports via `__all__` in ale-py so linting tools can better detect exports.
- Added builds for Python 3.11.

### Fixed

- Moved the Gym environment entrypoint from `gym.envs.atari:AtariEnv` to `ale_py.env.gym:AtariEnv`. This resolves many issues with the namespace package but does break backwards compatability for some Gym code that relied on the entry point being prefixed with `gym.envs.atari`.

## [0.7.5](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.7.4...v0.7.5) - 2022-04-18

### Added

- Added validation for Gym's frameskip values.
- Made ROM loading more robust with module-level `__getattr__` and `__dir__`.
- Added `py.typed` to the Python module's root directory to support type checkers.
- Bumped SDL to v2.0.16.

### Fixed

- Fixed Gym render mode metadata. (@vwxyzjn)
- Fixed Gym warnings about `seeding.hash_seed` and `random.randint`.
- Fixed build infrastructure issues from the migration to `setuptools>=0.61`.

### Removed

- Removed Gym's `.render(mode='human')`. Gym now uses the `render_mode` keyword argument in the environment constructor.

## [0.7.4](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.7.3...v0.7.4) - 2022-02-16

### Added

- Proper C++ namespacing for the ALE and Stella (@tuero)
- vcpkg manifest. You can now install dependencies via `cmake`.
- Support for the new Gym (0.22) `reset` API, i.e., the `seed` and `return_info` keyword arguments.
- Moved cibuildwheel config from Github Actions to pyproject.toml.

### Fixed

- Fixed a bug with the terminal signal in ChopperCommand #434
- Fixed warnings with `importlib-metadata` on Python < 3.9.
- Reverted the Gym `v5` defaults to align with the post-DQN literature. That is, moving from a frameskip of 5 -> 4, and full action set -> minimal action set.

## [0.7.3](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.7.3...v0.7.2) — 2021-11-02

### Added

- Environment variable `ALE_PY_ROM_DIR` which if specified will search for ROMs in `${ALE_PY_ROM_DIR}/*.bin`. (@joshgreaves)

## [0.7.2](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.7.1...v0.7.2) — 2021-10-07

### Added

- Package Tetris by Colin Hughes. This ROM is made publicly available by the author. This is useful for other open-source packages to be able to unit test agaisnt the ALE. (@tfboyd)
- Python 3.10 prebuilt wheels

### Fixed

- Fixed an issue with `isSupportedROM` on Windows which was causing incorrect ROM hashes.

### Removed

- Python 3.6 prebuilt wheels

## [0.7.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.7.0...v0.7.1) — 2021-09-28

### Added

- Added `ale-import-roms --import-from-pkg {pkg}`
- Use `gym.envs.atari` as a namespace package to maintain backwards compatability with the `AtariEnv` entry point.
- The ALE now uses Gym's environment plugin system in `gym>=0.21` (https://github.com/openai/gym/pull/2383, https://github.com/openai/gym/pull/2409, https://github.com/openai/gym/pull/2411). Users no longer are required to import `ale_py` to use a `-v5` environment.

### Changed

- Silence unsupported ROMs warning behind `ImportError`. To view these errors you should now supply the environment variable `PYTHONWARNINGS=default::ImportWarning:ale_py.roms`.
- Reworked ROM error messages to provide more helpful suggestions.
- General metadata changes to the Python package.

### Fixed

- Add missing `std::` name qualifier when enabling SDL (@anadrome)
- Fixed mandatory kwarg for `gym.envs.atari:AtariEnv.clone_state`.

## [0.7.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.6.1...v0.7.0) — 2021-09-14

### Added

- Native support for OpenAI Gym
- Native Python interface using pybind11 which results in a speedup for Python workloads as well as proper support for objects like `ALEState`
- Python ROM management, e.g., `ale-import-roms`
- PyPi Python wheels published as `ale-py` + we distribute SDL2 for out of the box visualization + audio support
- `isSupportedROM(path)` to check if a ROM file is supported by the ALE
- Added new games: Atlantis2, Backgammon, BasicMath, Blackjack, Casino, Crossbow, DarkChambers, Earthworld, Entombed, ET, FlagCapture, Hangman, HauntedHouse, HumanCannonball, Klax, MarioBros, MiniatureGolf, Othello, Pacman, Pitfall2, SpaceWar, Superman, Surround, TicTacToe3D, VideoCheckers, VideoChess, VideoCube, WordZapper (thanks @tkoppe)
- Added (additional) mode/difficulty settings for: Lost Luggage, Turmoil, Tron Dead Discs, Pong, Mr. Do, King Kong, Frogger, Adventure (thanks @tkoppe)
- Added `cloneState(include_rng)` which will eventually replace `cloneSystemState` (behind the scenes `cloneSystemState` is equivalent to `cloneState(include_rng=True)`).
- Added `setRAM` which can be useful for modifying the environment, e.g., learning a causal model over RAM transitions, altering game dynamics, etc.

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

## [0.6.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.6.0...v0.6.1) — 2019-11-20

### Changed

- Speedup of up to 30% by optimizing variable types (@qstanczyk)

### Fixed

- Fixed switch fall-through with Gravitar lives detection (@lespeholt)

## [0.6.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.5.2...v0.6.0) — 2015-06-23

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
- Fixed termination issues in Q\*Bert

## [0.5.2](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.5.1...v0.5.2) — 2015-10-04

### Added

- Routines for ALEState serialization (@Jragonmiris).

### Changed

- Enforce flags existence (@mcmachado).

### Fixed

- Fix RNG issues introduced in 0.5.0.
- Additional bug fixes.

## [0.5.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/v0.5.0...v0.5.1) — 2015-07-07

### Added

- Added RNG serialization capability.

### Changed

- Refactored Python getScreenRGB to return unpacked RGB values (@spragunr).
- Sets the default value of the color_averaging flag to be true. It was true by default in previous versions but was changed in 0.5.0. Reverted for backward compatibility.

### Fixed

- Bug fixes from ALE 0.5.0.

## [0.5.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/aa433a4b401bc3e7113c494edfc90500bc4afc78...v0.5.0) — 2015-06-22

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

## [0.4.4](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/d93189e0f00b5cb10120134ca965d8a5d3124581...aa433a4b401bc3e7113c494edfc90500bc4afc78) — 2014-04-28

### Fixed

- Fixed a memory issue in ALEScreen.

## [0.4.3](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/b905e07ead43d07f386b35128e7eff60595e1581...d93189e0f00b5cb10120134ca965d8a5d3124581) — 2014-04-26

### Fixed

- Fixed issues with frame numbers not being correctly updated.
- Fixed a bug where total reward was not properly reported under frame skipping.
- Fixed a bug with ALEState's m_frame_number.

## [0.4.2](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/ba33f16376b545462666268194e8f72df82c1a3a...b905e07ead43d07f386b35128e7eff60595e1581) — 2013-06-12

### Changed

- Modified StellaEnvironment save/load interface to provide additional flexibility.

### Fixed

- Series of bug fixes from Matthew Hausknecht and community.

## [0.4.1]https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/84f9678d713695314570e0f183072f36e177a364...ba33f16376b545462666268194e8f72df82c1a3a — 2013-05-24

### Added

- Added frame skipping support directly in StellaEnvironment.

### Changed

- Reverted default number of episodes to 10.

### Fixed

- Fixed RL-Glue syntax from OBSERVATION to OBSERVATIONS. Thanks to Angus MacIsaac for picking this bug up.

## [0.4.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/compare/5c45f643a78ef96ade23928fd6a3740172ec1e35...84f9678d713695314570e0f183072f36e177a364) — 2013-04-22

### Added

- RL-Glue support
- Shared library interface
- Simpler direct environment interfacing
- Improved environment handling
- Improved environment customization
- Better documentation

## 0.3.0 — 2012-07-22

- Initial ALE release.
