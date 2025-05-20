# ALE Vector Environment Guide

## Introduction

The Arcade Learning Environment (ALE) Vector Environment provides a high-performance implementation for running multiple Atari environments in parallel. This implementation utilizes native C++ code with multi-threading to achieve significant performance improvements, especially when running many environments simultaneously.

The vector environment is equivalent to `FrameStackObservation(AtariPreprocessing(gym.make("ALE/{AtariGame}-v5")), stack_size=4)`.

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
  - Next-step autoreset (see [blog](https://farama.org/Vector-Autoreset-Mode) for more detail)
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
from ale_py.vector_env import VectorAtariEnv

# Create a vector environment with 4 parallel instances of Breakout
envs = VectorAtariEnv(
    game="breakout",  # The ROM id not name, i.e., camel case compared to Gymnasium.make name versions
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
envs = VectorAtariEnv(
    # Required parameters
    game="breakout",          # The ROM id not name, i.e., camel case compared to Gymnasium.make name versions
    num_envs=8,               # Number of parallel environments

    # Preprocessing parameters
    frame_skip=4,             # Number of frames to skip (action repeat)
    grayscale=True,           # Use grayscale observations
    stack_num=4,              # Number of frames to stack
    img_height=84,            # Height to resize frames to
    img_width=84,             # Width to resize frames to
    maxpool=True,             # If to maxpool sequential frames

    # Environment behavior
    noop_max=30,              # Maximum number of no-ops at reset
    fire_reset=True,          # Press FIRE on reset for games that require it
    episodic_life=False,      # End episodes on life loss
    max_episode_steps=108000, # Max frames per episode (27000 steps * 4 frame skip)
    repeat_action_probability=0.0,  # Sticky actions probability
    full_action_space=False,  # Use full action space (not minimal)

    # Performance options
    batch_size=0,             # Number of environments to process at once (default=0 is the `num_envs`)
    num_threads=0,            # Number of worker threads (0=auto)
    thread_affinity_offset=-1,# CPU core offset for thread affinity (-1=no affinity)
    seed=0,                   # Random seed
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
envs = VectorAtariEnv(game="Breakout", num_envs=16, batch_size=4)
```

A smaller batch size can improve latency while a larger batch size can improve throughput.
When passing a batch size, the information will include the environment id of each observation
which is critical as the first (batch size) observations are returned for `reset` and `recv`.

### Thread Affinity

On systems with multiple CPU cores, setting thread affinity can improve performance:

```python
# Set thread affinity starting from core 0
envs = VectorAtariEnv(game="Breakout", num_envs=8, thread_affinity_offset=0)
```
