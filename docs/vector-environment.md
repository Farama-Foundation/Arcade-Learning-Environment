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

Additionally, with `grayscale=False` the shape is `(num_envs, stack_size, height, width, 3)` for RGB frames.

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

## PyTorch Integration

The vector environment can return PyTorch tensors via zero-copy custom ops using `env.torch()`. This is the PyTorch equivalent of `env.xla()` for JAX: no Python-side allocations occur in the hot path and the ops are compatible with `torch.compile`.

```python
handle_id, ale_send, ale_step, ale_recv, get_last_info, unregister = envs.torch()

# Single compiled step (send + recv fused)
obs, reward, term, trunc, steps_taken = ale_step(handle_id, actions)

# Or split for overlapping computation:
ale_send(handle_id, actions)
# ... do other work ...
obs, reward, term, trunc, steps_taken = ale_recv(handle_id)

# Info dict (lives, frame_number, etc.) from the last recv
info = get_last_info()

# Clean up when done
unregister()
```

Note: continous actions are not yet supported. 

## Multi-ROM Support

A single vector environment can run different ROMs in each slot by passing a list of game names. `num_envs` is inferred from the list length.

```python
envs = AtariVectorEnv(game=["breakout", "pong", "space_invaders", "qbert"])
```

When ROMs have different action set sizes, `single_action_space` is `None` and `action_space` is a `MultiDiscrete` with per-environment limits. Action bounds are enforced per environment; an out-of-range action raises an `IndexError`.

```python
# Per-environment action counts and sets
print(envs.num_actions)   # e.g. [4, 6, 6, 6]
print(envs.action_set)    # list[list[int]], one per env

# sample actions per environment as normal
actions = envs.action_space.sample()
```

## Multi-Step Action Sequences

Multi-step sequences support macro actions and open-loop options: the agent dispatches a sequence of primitive actions to be executed in the environment. Supporting lists of actions sent to each rom avoids unnecessary per-step Python to C++ overhead. 

Passing a list of arrays to `step` or `send` sends a sequence of actions to each environment in a single call.

```python
# Each array is the action sequence for one environment (lengths may differ)
sequences = [
    np.array([0, 2, 1, 0]),  # env 0: 4 actions
    np.array([3, 1]),         # env 1: 2 actions
]
obs, reward, term, trunc, info = envs.step(sequences)
print(info["steps_taken"])  # actions executed before early termination/trunction
```

`gamma` applies per-rom discount accumulation across the sequence. `paddle_strength` can be a scalar or a per-rom list.

```python
obs, reward, term, trunc, info = envs.step(sequences, gamma=0.99, paddle_strength=1.0)
```

An empty array `np.array([])` skips a rom entirely, returning its last observation with zero reward. In this instance, gamma should be 1 for this step. 
