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
