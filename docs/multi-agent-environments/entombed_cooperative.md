
# Emtombed: Cooperative

```{figure} ../_static/videos/multi-agent-environments/entombed_cooperative.gif
:width: 140px
:name: entombed_cooperative
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import entombed_cooperative_v3` |
|----------------------|--------------------------------------------------------|
| Actions              | Discrete                                               |
| Parallel API         | Yes                                                    |
| Manual Control       | No                                                     |
| Agents               | `agents= ['first_0', 'second_0']`                      |
| Agents               | 2                                                      |
| Action Shape         | (1,)                                                   |
| Action Values        | [0,17]                                                 |
| Observation Shape    | (210, 160, 3)                                          |
| Observation Values   | (0,255)                                                |
| Average Total Reward | 6.23                                                   |

Entombed's cooperative version is an exploration game where you need to work with your teammate to make it as far as possible into the maze. You both need to quickly navigate down a constantly generating maze you can only see part of. If you get stuck, you lose. Note you can easily find yourself in a dead-end escapable only through the use of rare power-ups. If players help each other by the use of these powerups, they can last longer. Note that optimal coordination requires that the agents be on opposite sides of the map, because powerups appear on one side or the other, but can be used to break through walls on both sides (the break is symmetric and effects both halves of the screen). In addition, there dangerous zombies lurking around to avoid.

The reward was designed to be identical to the single player rewards. In particular, an entombed stage is divided into 5 invisible sections. You receive reward immediately after changing sections, or after resetting the stage. Note that this means that you receive a reward when you lose a life, because it resets the stage, but not when you lose your last life, because the game terminates without the stage resetting.

[Official Entombed manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=165)

## Environment parameters

Environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

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
* v2: Breaking changes to entire API, fixed Entombed rewards (1.4.0)
* v1: Fixes to how all environments handle premature death (1.3.0)
* v0: Initial versions release (1.0.0)
