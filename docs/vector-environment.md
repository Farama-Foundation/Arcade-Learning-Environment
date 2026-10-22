# ALE Vector Environment Guide

## Introduction

The Arcade Learning Environment (ALE) Vector Environment provides a high-performance implementation for running multiple Atari environments in parallel. This implementation utilizes native C++ code with multi-threading to achieve significant performance improvements, especially when running many environments simultaneously.

The vector environment is equivalent to FrameStackObservation + AtariPreprocessing from Gymnasium as
```
gym_envs = gym.vector.SyncVectorEnv(
  [
      lambda: gym.wrappers.FrameStackObservation(
          gym.wrappers.AtariPreprocessing(
              gym.make(env_id, frameskip=1),
          ),
          stack_size=stack_num,
          padding_type="zero",
      )
      for _ in range(num_envs)
  ],
)
ale_envs = gym.make_vec(
  env_id,
  num_envs,
  use_fire_reset=False,
  reward_clipping=False,
  repeat_action_probability=0.0,
)
```

## Key Features

- **Parallel Execution**: Run multiple Atari environments simultaneously with minimal overhead
- **Standard Preprocessing**: Includes standard preprocessing steps from the Atari Deep RL literature:
  - Frame skipping
  - Observation resizing
  - Grayscale conversion
  - Frame stacking
  - NoOp initialization at reset
  - Fire reset (for games requiring the fire button to start)
  - Episodic life modes
- **Performance Optimizations**:
  - Native C++ implementation
  - Same-step and Next-step autoreset (see [blog](https://farama.org/Vector-Autoreset-Mode) for more detail)
  - Multi-threading for parallel execution
  - Thread affinity options for better performance on multi-core systems
  - Batch processing capabilities
- **Asynchronous Operation**: Split step operation into `send` and `recv` for more flexible control flow
- **Gymnasium Compatible**: Implements the Gymnasium `VectorEnv` [interface](https://gymnasium.farama.org/api/vector/)

## Installation

The vector implementation is packaged with ale-py that can be installed through PyPI, `pip install ale-py`.

Optionally, users can build the project locally, requiring VCPKG, that will install OpenCV to support observation preprocessing.

## Basic Usage

### Creating a Vector Environment

```python
from ale_py.vector_env import AtariVectorEnv

# Create a vector environment with 4 parallel instances of Breakout
envs = AtariVectorEnv(
    game="breakout",  # The ROM id not name, i.e., camel case compared to `gymnasium.make` name versions
    num_envs=4,
)

# Reset all environments
observations, info = envs.reset()

# Take random actions in all environments
actions = envs.action_space.sample()
observations, rewards, terminations, truncations, infos = envs.step(actions)

# Close the environment when done
envs.close()
```

## Advanced Configuration

The vector environment provides numerous configuration options:

```python
envs = AtariVectorEnv(
    # Required parameters
    game: str = "breakout",          # The ROM id not name, i.e., camel case compared to Gymnasium.make name versions
    num_envs: int = 1,               # Number of parallel environments
    *,

    # Preprocessing parameters
    frameskip: int = 4,             # Number of frames to skip (action repeat)
    grayscale: bool = True,         # Use grayscale observations
    stack_num: int = 4,             # Number of frames to stack
    img_height: int = 84,           # Height to resize frames to
    img_width: int = 84,            # Width to resize frames to
    maxpool: bool = True,           # If to maxpool sequential frames
    reward_clipping: bool = True,   # If to clip environment step rewards between -1 and 1

    # Environment behavior
    noop_max: int = 30,             # Maximum number of no-ops at reset
    use_fire_reset: bool = True,    # Press FIRE on reset for games that require it
    episodic_life: bool = False,    # End episodes on life loss
    life_loss_info: bool = False,   # Return termination signal on life loss but don't reset the environment until all lives are alot. If used, this MUST be indicated as has a significant impact on training performance.
    max_num_frames_per_episode: int = 108000, # Max frames per episode (27000 steps * 4 frame skip)
    repeat_action_probability: float = 0.0,   # Sticky actions probability
    full_action_space: bool = False,          # Use full action space (not minimal)
    continuous: bool = False,                 # If to use continuous actions
    continuous_action_threshold: bool = 0.5,  # The threshold at which to use continuous actions
    reset_pool_size: int = 8,                 # Start states captured from real resets and replayed on later resets (0 = emulate every reset)
    reset_pool_prefill: bool = False,         # Capture all reset_pool_size start states on the first reset instead of over the first episodes

    # Performance options
    batch_size=0,             # Number of environments to process at once (default=0 is the `num_envs`)
    autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,  # How reset sub-environments when they terminated (https://farama.org/Vector-Autoreset-Mode)
    num_threads=0,            # Number of worker threads (0=auto)
    thread_affinity_offset=-1,# CPU core offset for thread affinity (-1=no affinity)
)
```

## Observation Format

The observation format from the vector environment is:

```
observations.shape = (num_envs, stack_size, height, width)
```

Where:
- `num_envs`: Number of parallel environments
- `stack_size`: Number of stacked frames (typically 4)
- `height`, `width`: Image dimensions (typically 84x84)

Additionally, with `grayscale=True` then the shape is `(num_envs, stack_size, height, width, 3)` for RGB frames.

## Performance Considerations

### Number of Environments

Increasing the number of environments typically improves throughput until you hit CPU core limits.
For optimal performance, set `num_envs` close to the number of physical CPU cores.

### Send/Recv vs Step

Using the `send`/`recv` API can allow for better overlapping of computation and environment stepping:

```python
# Send actions to environments
envs.send(actions)

# Do other computation here while environments are stepping

# Receive results when ready
observations, rewards, terminations, truncations, infos = envs.recv()
```

### Batch Size

The `batch_size` parameter controls how many environments are processed simultaneously by the worker threads:

```python
# Process environments in batches of 4
envs = AtariVectorEnv(game="Breakout", num_envs=16, batch_size=4)
```

A smaller batch size can improve latency while a larger batch size can improve throughput.
When passing a batch size, the information will include the environment id of each observation
which is critical as the first (batch size) observations are returned for `reset` and `recv`.

### Thread Affinity

On systems with multiple CPU cores, setting thread affinity can improve performance:

```python
# Set thread affinity starting from core 0
envs = AtariVectorEnv(game="Breakout", num_envs=8, thread_affinity_offset=0)
```

### Reset Pool

Resetting an Atari console emulates around 68 frames of bring-up, which costs 10–16× a step.
Each sub-environment therefore emulates its first `reset_pool_size` resets for real, keeps a snapshot of each start state,
and on later resets restores a snapshot picked uniformly at random instead.

A pool rather than a single snapshot is needed because a reset is not a pure function of the ROM:
the console's timer is redrawn on every reset, some cartridges re-randomise their RAM, and console RAM carries over between episodes.
Every start state reachable through the pool is one a real reset produced.
A seeded `reset(seed=...)` reloads the ROM and empties the pool.

```python
# Capture all 8 start states on the first reset, so every later episode boundary costs the same
envs = AtariVectorEnv(game="breakout", num_envs=8, reset_pool_prefill=True)

# Emulate every reset, as versions before the pool did
envs = AtariVectorEnv(game="breakout", num_envs=8, reset_pool_size=0)
```

`reset_pool_prefill` is off by default because it captures its start states from consecutive resets with no gameplay between them.
On ROMs whose start state depends on the console RAM left over by the previous episode, it sees fewer distinct start states
(for example, `yars_revenge` reaches 8 distinct start states with the default and 1 with prefill).
