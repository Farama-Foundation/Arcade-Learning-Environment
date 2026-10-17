---
title: Atari Environments
firstpage:
---

# Multi-agent environments

```{toctree}
:hidden:
multi-agent-environments/basketball_pong
multi-agent-environments/boxing
multi-agent-environments/combat_plane
multi-agent-environments/combat_tank
multi-agent-environments/double_dunk
multi-agent-environments/entombed_competitive
multi-agent-environments/entombed_cooperative
multi-agent-environments/flag_capture
multi-agent-environments/foozpong
multi-agent-environments/ice_hockey
multi-agent-environments/joust
multi-agent-environments/mario_bros
multi-agent-environments/maze_craze
multi-agent-environments/othello
multi-agent-environments/pong
multi-agent-environments/quadrapong
multi-agent-environments/space_invaders
multi-agent-environments/space_war
multi-agent-environments/surround
multi-agent-environments/tennis
multi-agent-environments/video_checkers
multi-agent-environments/volleyball_pong
multi-agent-environments/warlords
multi-agent-environments/wizard_of_wor
```

The Atari environments are based off the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment). This environment was instrumental in the development of modern reinforcement learning, and so we hope that our [multi-agent version](https://github.com/Farama-Foundation/Multi-Agent-ALE) of it will be useful in the development of multi-agent reinforcement learning.

The multiplayer support from Multi-Agent-ALE is now part of the ALE itself. The environments below are provided by [PettingZoo](https://pettingzoo.farama.org/environments/atari/); libraries that want to build their own multi-agent API can use the {ref}`ALEInterface multiplayer methods <ale-multiplayer-interface>` directly.

```{raw} html
    :file: multi-agent-environments/list.html
```

## Installation

The unique dependencies for this set of environments can be installed via:

````bash
pip install pettingzoo[atari]
````

## Usage

To launch a [Space Invaders](https://pettingzoo.farama.org/environments/multi-agent-environments/space_invaders/) environment with agents taking random actions:
``` python
from pettingzoo.atari import space_invaders_v2
env = space_invaders_v2.env(render_mode="human")

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        env.action_space(agent).sample()  # this is where you would insert your policy
    env.step(action)
env.close()
```

## Games Overview

Most games have two players, except for Warlords and a couple of Pong variations which have four players.

## Environment Details

The ALE has been studied extensively and a few notable problems have been identified:

* **Determinism**: The Atari console is deterministic, and so agents can theoretically memorize precise sequences of actions that will maximize the end score. This is not ideal, so we encourage the use of [SuperSuit's](https://github.com/Farama-Foundation/SuperSuit) `sticky_actions` wrapper (example below). This is the recommended approach of  *"Machado et al. (2018), "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents"*
* **Frame flickering**: Atari games often do not render every sprite every frame due to hardware restrictions. Instead, sprites (such as the knights in Joust) are sometimes rendered every other frame or even (in Wizard of Wor) every 3 frames. The standard way of handling this computing the pixel-wise maximum of the previous 2 observations (see example below for implementation).

## Preprocessing

We encourage the use of the [supersuit](https://github.com/Farama-Foundation/SuperSuit) library for preprocessing. The unique dependencies for this set of environments can be installed via:

 ````bash
pip install supersuit
 ````

Here is some example usage for the Atari preprocessing:

``` python
import supersuit
from pettingzoo.atari import space_invaders_v1

env = space_invaders_v1.env()

# as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
# to deal with frame flickering
env = supersuit.max_observation_v0(env, 2)

# repeat_action_probability is set to 0.25 to introduce non-determinism to the system
env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

# skip frames for faster processing and less control
# to be compatible with gym, use frame_skip(env, (2,5))
env = supersuit.frame_skip_v0(env, 4)

# downscale observation for faster processing
env = supersuit.resize_v1(env, 84, 84)

# allow agent to see everything on the screen despite Atari's flickering screen problem
env = supersuit.frame_stack_v1(env, 4)
```

## Common Parameters

All the Atari environments have the following environment parameters:

``` python
<atari_game>.env(obs_type='rgb_image', full_action_space=True, max_cycles=100000, auto_rom_install_path=None)
```

`obs_type`:  There are three possible values for this parameter:

* 'rgb_image' (default) - produces an RGB image like you would see as a human player.
* 'grayscale_image' - produces a grayscale image.
* 'ram' - produces an observation of the 1024 bits that comprise the RAM of the Atari console.

`full_action_space`: Setting this option to True sets the action space to the full space of 18 action. Setting this to `False` (default) removes duplicate actions and leaves only unique actions.

`max_cycles`:  the number of frames (the number of steps that each agent can take) until game terminates.

`auto_rom_install_path`: The path to your AutoROM installation, installed
with the [Farama-Foundation/AutoROM](https://github.com/Farama-Foundation/AutoROM) tool.
This is the path you specified when installing AutoROM. For example, if
you're using the boxing Atari environment, then the library will look for
the rom at
`/auto_rom_install_path/ROM/boxing/boxing.bin`.
If this is not specified (has value `None`), then the library looks for roms
installed at the default AutoROM path.

(ale-multiplayer-interface)=

## Using the ALEInterface directly

`ale_py.ALEInterface` (and the C++ `ale::ALEInterface`) supports multiplayer game modes directly, so other libraries can implement their own multi-agent environment API. The single-player methods are unchanged; the multiplayer variants are overloads of them, or new methods where an overload isn't possible.

```python
import numpy as np
from ale_py import ALEInterface, roms

ale = ALEInterface()
ale.setInt("random_seed", 123)
ale.loadROM(roms.get_rom_path("pong"))

# The game mode selects the number of players
mode = ale.getAvailableModes(num_players=2)[0]
ale.setMode(mode)
ale.reset_game()
assert ale.numPlayersActive() == 2

actions = ale.getMinimalActionSet()
total_rewards = np.zeros(2)
while not ale.game_over():
    # One action per player, ordered by player number
    rewards = ale.act([np.random.choice(actions), np.random.choice(actions)])
    total_rewards += rewards
    lives = ale.allLives()  # one entry per player

print(f"Episode ended with scores: {total_rewards}")
```

### Methods

| Method | Description |
|--------|-------------|
| `getAvailableModes(num_players)` | The modes for 1, 2, 3 or 4 players. `getAvailableModes()` (no argument) still returns only the single-player modes. |
| `setMode(mode)` | Accepts a mode for any number of players. As for single-player modes, the change takes effect at the next `reset_game()`. |
| `numPlayersActive()` | The number of players in the current mode, which is the number of actions `act` expects. |
| `act(actions, paddle_strengths=None)` | Takes one action per active player and returns a list of one reward per player. `paddle_strengths` optionally gives a paddle strength per player for continuous actions (default `1.0`). Accepts lists or NumPy arrays. |
| `allLives()` | One lives value per active player (see below). |

Things to be aware of:

* **Actions** use the player A action range (`0`–`17`, as returned by `getMinimalActionSet()` / `getLegalActionSet()`) for every player; a player is identified by its position in the list. Passing a `PLAYER_B_*` action raises an error.
* **Single-player methods**: `act(action)` and `lives()` raise a `RuntimeError` in multiplayer modes. In single-player modes `act([action], [strength])` returns `[act(action, strength)]` and is identical to it, including the sticky-action random stream, so a library can use the list form for every mode.
* **Termination** is game-wide: `game_over()` and `game_truncated()` apply to all players. A player whose `allLives()` entry is negative has been eliminated while the game continues (e.g. Warlords). Otherwise, player 1 reports the game's lives counter and players 2–4 report `1` while the game is running and `0` after it ends.
* **Observations** (`getScreenRGB()`, `getScreenGrayscale()`, `getRAM()`, `getAudio()`) are shared by all players.
* **Sticky actions**: `repeat_action_probability` is applied to each player independently.
* **Controllers**: joystick games support at most 2 players; paddle games (Pong, Warlords) support up to 4.
* **Default mode**: Combat and Maze Craze have no single-player modes, so they start in a two-player mode after `loadROM`; all other games start in a single-player mode.
* **Stalling**: in the turn-based two-player modes of Double Dunk, Othello, Tennis and Video Checkers, a player who doesn't act within a time limit is penalised. The limit can be set in frames with `ale.setInt("max_turn_time", frames)` before `loadROM`; `-1` (the default) uses each game's own limit.
* **Vector environments** (`gym.make_vec` / `AtariVectorEnv`) only support single-player modes.

### Modes by number of players

No game has three-player modes.

| ROM | 1 player | 2 players | 4 players |
|-----|----------|-----------|-----------|
| backgammon | 0 | 4 | — |
| boxing | 0 | 2 | — |
| combat | — | 1–27 | — |
| double_dunk | 0–15 | 16–31 | — |
| entombed | 0 | 2–3 | — |
| fishing_derby | 0 | 2 | — |
| flag_capture | 8–10 | 1–4 | — |
| ice_hockey | 0, 2 | 1, 3 | — |
| joust | 0, 2 | 1, 3 | — |
| mario_bros | 0, 2, 4, 6 | 1, 3, 5, 7 | — |
| maze_craze | — | 0–63 | — |
| othello | 0–2 | 3 | — |
| pong | 0–1 | 2–3, 8–9, 12–13, 18–19, 22–27, 34–35, 38–39, 42–45 | 4–7, 10–11, 14–17, 20–21, 28–33, 36–37, 40–41, 46–49 |
| space_invaders | 0–15 | 32–63 | — |
| space_war | 6–17 | 1–5 | — |
| surround | 0, 2 | 1, 5–12 | — |
| tennis | 0, 2 | 1, 3 | — |
| video_checkers | 1–9, 11–19 | 10 | — |
| warlords | 4, 9, 14, 19 | 3, 5, 8, 10, 13, 15, 18, 20, 23 | 1, 6, 11, 16, 21 |
| wizard_of_wor | 0 | 1 | — |

```{note}
Pong's mode numbers are one lower than in Multi-Agent-ALE, so that its single-player modes keep their existing ALE numbers (`0` and `1`). Code that hard-codes Multi-Agent-ALE Pong modes needs to subtract one.
```


## Citation

Multiplayer games within the Arcade Learning Environment were introduced in:

```
@article{terry2020arcade,
  Title = {Multiplayer Support for the Arcade Learning Environment},
  Author = {Terry, J K and Black, Benjamin},
  journal={arXiv preprint arXiv:2009.09341},
  year={2020}
}
```

The Arcade Learning Environment was originally introduced in:

```
@Article{bellemare13arcade,
  author = { {Bellemare}, M.~G. and {Naddaf}, Y. and {Veness}, J. and {Bowling}, M.},
  title = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
  journal = {Journal of Artificial Intelligence Research},
  year = "2013",
  month = "jun",
  volume = "47",
  pages = "253--279",
}
```

Various extensions to the Arcade Learning Environment were introduced in:

```
@article{machado2018revisiting,
  title={Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents},
  author={Machado, Marlos C and Bellemare, Marc G and Talvitie, Erik and Veness, Joel and Hausknecht, Matthew and Bowling, Michael},
  journal={Journal of Artificial Intelligence Research},
  volume={61},
  pages={523--562},
  year={2018}
}
```
