# Python Interface

Aside from a few minor differences, the Python interface mirrors the C++ interface. For example, the following implements a random agent:

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

## ROM Support

The Python interface introduces some nice-to-have tools for managing ROMs. Specifically we provide the command line tool `ale-import-roms`. By passing a directory argument to this command you can simply import all supported ROMs from the directory. For example,

```shell
$ ale-import-roms roms/

[SUPPORTED]       breakout        roms/breakout.bin

Imported 1/1 ROMs
```

Once you've imported ROMs you will be able to import the ROM as follows:

```py
import ale_py
from ale_py.roms import Breakout # Note; ROMs are camelcase

ale = ale_py.ALEInterface()
ale.loadROM(Breakout)
```

As a more advanced feature the ALE provides the entry point `ale-py.roms` where other Python packages can register ROMs for discovery. To see a more detailed example check out [Arcade-Learning-Environment/examples/python-rom-package](https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/examples/python-rom-package).
