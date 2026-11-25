# ALE Vector Environment Guide

## Introduction

The Arcade Learning Environment (ALE) Vector Environment provides a high-performance implementation for running multiple Atari environments in parallel. This implementation utilizes native C++ code with multi-threading to achieve significant performance improvements, especially when running many environments simultaneously.

The vector environment is equivalent to Gymnasium's `FrameStackObservation` + `AtariPreprocessing` wrappers, i.e., the following two are equivalent:

```python
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
- **JAX / XLA Support**: Call `reset` and `step` from jitted JAX code (see the JAX / XLA section below)
- **Gymnasium Compatible**: Implements the Gymnasium `VectorEnv` [interface](https://gymnasium.farama.org/api/vector/)

## Installation

The vector implementation is packaged with `ale-py`, which can be installed through PyPI with `pip install ale-py`.

Optionally, users can build the project locally, which requires vcpkg to install OpenCV for observation preprocessing.

## Basic Usage

### Creating a Vector Environment

```python
from ale_py.vector_env import AtariVectorEnv

# Create a vector environment with 4 parallel instances of Breakout
envs = AtariVectorEnv(
    game="breakout",  # The ROM id (snake_case), not the `gymnasium.make` name (e.g., "space_invaders" not "SpaceInvaders")
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
    game: str = "breakout",          # The ROM id (snake_case), not the `gymnasium.make` name
    num_envs: int = 1,               # Number of parallel environments (required)
    *,

    # Preprocessing parameters
    frameskip: int = 4,             # Number of frames to skip (action repeat)
    grayscale: bool = True,         # Use grayscale observations
    stack_num: int = 4,             # Number of frames to stack
    img_height: int = 84,           # Height to resize frames to
    img_width: int = 84,            # Width to resize frames to
    maxpool: bool = True,           # Whether to max-pool sequential frames
    reward_clipping: bool = True,   # Whether to clip environment step rewards between -1 and 1

    # Environment behavior
    noop_max: int = 30,             # Maximum number of no-ops at reset
    use_fire_reset: bool = True,    # Press FIRE on reset for games that require it
    episodic_life: bool = False,    # End episodes on life loss
    life_loss_info: bool = False,   # Return termination signal on life loss but don't reset the environment until all lives are lost. If used, this MUST be reported as it has a significant impact on training performance.
    max_num_frames_per_episode: int = 108000, # Max frames per episode (27000 steps * 4 frame skip)
    repeat_action_probability: float = 0.0,   # Sticky actions probability
    full_action_space: bool = False,          # Use full action space (not minimal)
    continuous: bool = False,                 # Whether to use continuous actions
    continuous_action_threshold: float = 0.5, # The threshold above which a continuous action triggers a discrete event
    reset_pool_size: int = 256,               # Start states captured from real resets and replayed on later resets, split between the sub-environments (0 = emulate every reset)
    reset_pool_prefill: bool = False,         # Capture each sub-environment's share of the pool on the first reset instead of over its first episodes

    # Performance options
    batch_size=0,             # Number of environments to process at once (default=0 is the `num_envs`)
    autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,  # How to reset sub-environments when they terminate (https://farama.org/Vector-Autoreset-Mode)
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

With `grayscale=False`, the shape is `(num_envs, stack_size, height, width, 3)` for RGB frames.

## Seeding and Partial Resets

`reset` accepts either an integer seed, which seeds the sub-environments with consecutive values starting at `seed`, or a NumPy array with one seed per sub-environment being reset.

Individual sub-environments can be reset with the `reset_mask` option, a boolean array of shape `(num_envs,)`. Sub-environments that are masked out are left untouched and report their current observation and info. Partial resets require `batch_size == num_envs` and a preceding full reset.

```python
import numpy as np

observations, info = envs.reset(seed=np.array([10, 20, 30, 40]))

# Reset only the first and third sub-environments, seeding them with 1 and 2
reset_mask = np.array([True, False, True, False])
observations, info = envs.reset(seed=np.array([1, 2]), options={"reset_mask": reset_mask})
```

## JAX / XLA Integration

The vector environment can be called directly from [JAX](https://github.com/jax-ml/jax), including inside `jax.jit` and `jax.lax.scan`, through XLA foreign function calls. This avoids leaving the JAX program to step the environments. The XLA support requires `jax` and `chex`, which can be installed with:

```bash
pip install "ale-py[xla]"
```

`envs.xla()` returns a handle to the environments along with pure `reset` and `step` functions. Each function takes the current handle and returns a new handle that must be passed to the next call, which keeps the order of environment calls explicit to XLA.

```python
import jax
import jax.numpy as jnp
import numpy as np
from ale_py import AtariVectorEnv

envs = AtariVectorEnv("pong", num_envs=4)
handle, xla_reset, xla_step = envs.xla()

handle, (observations, info) = xla_reset(handle, seed=np.arange(4))

@jax.jit
def step(handle, actions):
    return xla_step(handle, actions)

# Roll out 100 steps with random actions inside a single jitted scan
actions = jax.random.randint(
    jax.random.key(0), (100, 4), 0, envs.single_action_space.n, dtype=jnp.int32
)
handle, (observations, rewards, terminations, truncations, info) = jax.lax.scan(
    step, handle, xs=actions
)
envs.close()
```

`xla_reset(handle, seed=None, reset_mask=None)` supports the same seeding and partial resets as `reset`, and `xla_step(handle, actions)` expects `int32` actions of shape `(batch_size,)` (or `float32` actions of shape `(batch_size, 3)` with `continuous=True`).

Notes:
- Only `autoreset_mode=AutoresetMode.NEXT_STEP` (the default) is supported.
- CPU handlers are available on all platforms. GPU (CUDA) handlers are included in the Linux and Windows wheels so the functions can be used within GPU programs; the environments themselves still run on the CPU.
- The handle refers to the Python `AtariVectorEnv` object, so the object must be kept alive (and not closed) while the XLA functions are used.

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
envs = AtariVectorEnv(game="breakout", num_envs=16, batch_size=4)
```

A smaller batch size can improve latency while a larger batch size can improve throughput.
When `batch_size` is smaller than `num_envs`, `reset` and `recv` return the first `batch_size` environments to finish,
so the info includes `env_id`, identifying which environment each observation belongs to.

### Thread Affinity

On systems with multiple CPU cores, setting thread affinity can improve performance:

```python
# Set thread affinity starting from core 0
envs = AtariVectorEnv(game="breakout", num_envs=8, thread_affinity_offset=0)
```

### Reset Pool

Resetting an Atari console emulates around 68 frames of bring-up, which costs 10–16× a step.
`reset_pool_size` start states are therefore split evenly between the sub-environments (at least one each).
Each sub-environment emulates its share of resets for real, keeps a snapshot of each start state,
and on later resets restores one of its own snapshots, picked uniformly at random, instead.
Snapshots are not shared between sub-environments, so the total number of distinct start states is the same whatever `num_envs` is.

The pool size does not change the steady-state steps per second, because a draw costs the same however many snapshots there are.
A larger pool gives more distinct start states, but each sub-environment emulates more resets before it reaches the steady state,
and fewer sub-environments means a larger share each (with the default of 256, a single environment emulates its first 256 resets).
Each snapshot takes about 6 KB.

A pool rather than a single snapshot is needed because a reset is not a pure function of the ROM:
the console's timer is redrawn on every reset, some cartridges re-randomise their RAM, and console RAM carries over between episodes.
Every start state reachable through the pool is one a real reset produced.
A seeded `reset(seed=...)` reloads the ROM and empties the pool.

```python
# Capture all 256 start states (32 per sub-environment) on the first reset, so every later episode boundary costs the same
envs = AtariVectorEnv(game="breakout", num_envs=8, reset_pool_prefill=True)

# Emulate every reset, as versions before the pool did
envs = AtariVectorEnv(game="breakout", num_envs=8, reset_pool_size=0)
```

`reset_pool_prefill` is off by default because it captures its start states from consecutive resets with no gameplay between them.
On ROMs whose start state depends on the console RAM left over by the previous episode, it sees fewer distinct start states
(for example, with 8 start states per sub-environment, `yars_revenge` reaches 8 distinct start states without prefill and 1 with it).

#### Reset Distribution

How many start states a pool needs depends on the ROM. The table counts the distinct start states `reset_game()` produced over 10,000 resets per ROM, at sticky-action probabilities 0 and 0.25; the counts were the same at both. Two start states count as distinct when a fixed 100-step action script played from them gives different frames or rewards. Every count is therefore a lower bound on what an agent can see.

| ROM | Distinct start states | Depends on the previous episode | Probability held by a 256-state pool | ... by a 1,024-state pool |
|---|---:|:-:|---:|---:|
| `elevator_action` | 1,870, not saturated (about 9,700 estimated) | no | 0.56 | 0.76 |
| `yars_revenge` | 214 | yes | 0.72 | 0.98 |
| `sir_lancelot` | 64 | yes | 0.97 | 1.00 |
| `solaris` | 62 | no | 0.98 | 1.00 |
| `assault` | 58 | no | 0.98 | 1.00 |
| `surround` | 39 | yes | 0.99 | 1.00 |
| `ice_hockey` | 27 | yes | 0.99 | 1.00 |
| `haunted_house` | 21 | yes | 0.99 | 1.00 |
| `space_war` | 4 | no | 1.00 | 1.00 |
| `hangman` | 3 | yes | 1.00 | 1.00 |

Two start states (14 ROMs): `beam_rider`, `boxing`, `chopper_command`, `demon_attack`, `fishing_derby`, `freeway`, `frostbite`, `hero`, `kaboom`, `pitfall2`, `private_eye`, `seaquest`, `tennis` and `time_pilot`. Every other ROM (80) has a single start state, so any pool size reproduces it exactly. (`double_dunk` showed 14 start states at sticky 0.25 until #765 kept sticky actions out of its options menu.)

The probability columns give the expected share of real resets whose start state a pool of that many draws contains. A pool reproduces `reset_game()` well when this share is close to 1, which with the default of 256 holds for every ROM except two:

* `elevator_action` varies only in emulator state that is neither RAM nor screen (the console timer). It has thousands of start states, and about one reset in seven still produced a new one after 10,000 resets. Use `reset_pool_size=0` if you need its exact start-state distribution.
* `yars_revenge` has 214 start states, and a pool of 1,024 holds 98% of them by probability.

For ROMs that depend on the previous episode, the start state depends on how that episode was played. The pool captures its start states during the first episodes of training and keeps them for the rest of the run, so later episodes start from states the early policy left behind.

To reproduce the table, or measure another ROM or configuration:

```bash
python scripts/count-reset-states.py elevator_action solaris yars_revenge --resets 1000 --jobs 8
python scripts/count-reset-states.py --all --resets 10000 --jobs 22 --cache reset-cache --json resets.json
```
