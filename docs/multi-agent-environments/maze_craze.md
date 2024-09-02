
# Maze Craze

```{figure} ../_static/videos/multi-agent-environments/maze_craze.gif
:width: 140px
:name: maze_craze
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import maze_craze_v3` |
|----------------------|----------------------------------------------|
| Actions              | Discrete                                     |
| Parallel API         | Yes                                          |
| Manual Control       | No                                           |
| Agents               | `agents= ['first_0', 'second_0']`            |
| Agents               | 2                                            |
| Action Shape         | (1,)                                         |
| Action Values        | [0,17]                                       |
| Observation Shape    | (250, 160, 3)                                |
| Observation Values   | (0,255)                                      |

A competitive game of memory and planning!Its a race to leave the maze. There are 3 main versions of the game.

1. **Race**: A basic version of the game. First to leave the maze wins
2. **Robbers**: There are 2 robbers randomly traversing the maze. If you are captured by the robbers, you lose the game, and receive -1 reward, and will be done. The player that has not been captured will not receive any reward, but they can still exit the maze and win, scoring +1 reward.
3. **Capture**: Each player have to capture all 3 robbers before you are able to exit the maze. Additionally, you can confuse your opponent (and yourself, if you are not careful!) by creating a block that looks identical to a wall in the maze, but all players can pass through it. You can only
create one wall at a time, when you create a new one, the old one disappears.

The first player to leave the maze scores +1, the other player scores -1 (unless that other player has already been captured in Robbers mode).

[Official Maze craze manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=295). Note that the table of modes has some inaccuracies. In particular, game mode 12 has Blockade enabled, not mode 11.

## Environment parameters

Some environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

Parameters specific to Maze Craze are

``` python
maze_craze.env(game_version="robbers", visibilty_level=0)
```

* `game_version`:  Possibilities are "robbers", "race", "capture", corresponding to the 3 game versions described above
* `visibilty_level`:  A number from 0-3. Set to 0 for 100% visible map, and 3 for 0% visibility map.

## Action Space

In any given turn, an agent can choose from one of 18 actions.

| Value   | Meaning      | Value   | Meaning         | Value   | Meaning        |
|---------|--------------|---------|-----------------|---------|----------------|
| `0`     | `NOOP`       | `1`     | `FIRE`          | `2`     | `UP`           |
| `3`     | `RIGHT`      | `4`     | `LEFT`          | `5`     | `DOWN`         |
| `6`     | `UPRIGHT`    | `7`     | `UPLEFT`        | `8`     | `DOWNRIGHT`    |
| `9`     | `DOWNLEFT`   | `10`    | `UPFIRE`        | `11`    | `RIGHTFIRE`    |
| `12`    | `LEFTFIRE`   | `13`    | `DOWNFIRE`      | `14`    | `UPRIGHTFIRE`  |
| `15`    | `UPLEFTFIRE` | `16`    | `DOWNRIGHTFIRE` | `17`    | `DOWNLEFTFIRE` |

See [environment specification](../env-spec) to see more information on the action meaning.


## Version History

* v3: Minimal Action Space (1.18.0)
* v2: Breaking changes to entire API (1.4.0)
* v1: Fixes to how all environments handle premature death (1.3.0)
* v0: Initial versions release (1.0.0)
