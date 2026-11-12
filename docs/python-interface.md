# Python Interface

Aside from a few minor differences, the Python interface mirrors the [C++ interface](cpp-interface.md). For example, the following implements a random agent:

```python
import sys
from random import randrange
from ale_py import ALEInterface

def main(rom_file):
    ale = ALEInterface()
    ale.setInt('random_seed', 123)
    ale.loadROM(rom_file)

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()
    num_actions = len(legal_actions)

    total_reward = 0
    while not ale.game_over():
      a = legal_actions[randrange(num_actions)]
      reward = ale.act(a)
      total_reward += reward

    print(f'Episode ended with score: {total_reward}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
      print(f"Usage: {sys.argv[0]} rom_file")
      sys.exit()

    rom_file = sys.argv[1]
    main(rom_file)
```

The available setting keys for `setBool`, `setInt`, `setFloat` and `setString` are listed in the [environment specification](env-spec.md#settings).

## ROMs

The supported ROMs are bundled with `ale-py`. Their paths can be found with `ale_py.roms`, using the snake_case ROM id:

```python
from ale_py import ALEInterface, roms

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("breakout"))

print(roms.get_all_rom_ids())  # ['adventure', 'air_raid', 'alien', ...]
```

To load ROMs from another directory, set the `ALE_ROMS_DIR` environment variable.

## Observations

The screen can be read as an RGB array of shape `(210, 160, 3)` with `getScreenRGB`, as a grayscale array of shape `(210, 160)` with `getScreenGrayscale`, or as the raw palette indices with `getScreen`. The console's 128 bytes of RAM are available with `getRAM`. Each of these methods optionally takes a preallocated NumPy array to write into, avoiding an allocation on every step.

```python
import numpy as np

rgb = ale.getScreenRGB()
ram = ale.getRAM()

# Reuse the same buffer every step
buffer = np.empty((210, 160), dtype=np.uint8)
ale.getScreenGrayscale(buffer)
```

## Saving and Restoring States

`cloneState` returns an `ALEState` snapshot of the emulator, which can later be passed to `restoreState`. By default, the snapshot does not include the environment's random number generator (used for sticky actions), making it suitable for planning; pass `include_rng=True` to also capture it so that restoring reproduces the exact same future.

```python
state = ale.cloneState()
for _ in range(10):
    ale.act(ale.getLegalActionSet()[0])
ale.restoreState(state)  # back to where we started
```

`ALEState` objects can be pickled or converted to bytes with `serialize()`, and recreated with `ALEState(serialized)`. Serialized states are not guaranteed to be compatible across ALE versions.

## API Reference

```{eval-rst}
.. autoclass:: ale_py.ALEInterface
    :members:
    :undoc-members:

.. autoclass:: ale_py.ALEState
    :members:
    :undoc-members:

.. autoclass:: ale_py.Action
    :members:
    :undoc-members:

.. autoclass:: ale_py.LoggerMode
    :members:
    :undoc-members:
```
