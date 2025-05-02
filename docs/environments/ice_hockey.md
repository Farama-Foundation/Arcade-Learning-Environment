---
title: IceHockey
---

# IceHockey

```{figure} ../_static/videos/environments/ice_hockey.gif
:width: 120px
:name: IceHockey
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                    |
|-------------------|------------------------------------|
| Make              | gymnasium.make("ALE/IceHockey-v5") |
| Action Space      | Discrete(18)                       |
| Observation Space | Box(0, 255, (210, 160, 3), uint8)  |

For more IceHockey variants with different observation and action spaces, see the variants section.

## Description

Your goal is to score as many points as possible in a standard game of Ice Hockey over a 3-minute time period. The ball is usually called "the puck". There are 32 shot angles ranging from the extreme left to the extreme right. The angles can only aim towards the opponent's goal.Just as in real hockey, you can pass the puck by shooting it off the sides of the rink. This can be really key when you're in position to score a goal.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=241)

## Actions

IceHockey has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As IceHockey uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

|   Value | Meaning       |
|---------|---------------|
|       0 | NOOP          |
|       1 | FIRE          |
|       2 | UP            |
|       3 | RIGHT         |
|       4 | LEFT          |
|       5 | DOWN          |
|       6 | UPRIGHT       |
|       7 | UPLEFT        |
|       8 | DOWNRIGHT     |
|       9 | DOWNLEFT      |
|      10 | UPFIRE        |
|      11 | RIGHTFIRE     |
|      12 | LEFTFIRE      |
|      13 | DOWNFIRE      |
|      14 | UPRIGHTFIRE   |
|      15 | UPLEFTFIRE    |
|      16 | DOWNRIGHTFIRE |
|      17 | DOWNLEFTFIRE  |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the q"rgb" type

See variants section for the type of observation used by each environment id by default.

### Reward

You score points by shooting the puck into your opponent's goal. Your opponent scores in the same manner. There are no limits to how many points you can get per game, other than the time limit of 3-minute games. For a more detailed documentation, consult [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=241).

## Variants

IceHockey has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                  | obs_type=   | frameskip=   | repeat_action_probability=   |
|-------------------------|-------------|--------------|------------------------------|
| IceHockey-v0            | `rgb`       | `(2, 5)`     | `0.25`                       |
| IceHockeyNoFrameskip-v0 | `rgb`       | `1`          | `0.25`                       |
| IceHockey-v4            | `rgb`       | `(2, 5)`     | `0.00`                       |
| IceHockeyNoFrameskip-v4 | `rgb`       | `1`          | `0.00`                       |
| ALE/IceHockey-v5        | `rgb`       | `4`          | `0.25`                       |

See the [version history page](https://ale.farama.org/environments/#version-history-and-naming-schemes) to implement previously implemented environments, e.g., `IceHockeyNoFrameskip-v4`.

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, 2]`          | `0`            | `[0, 1, 2, 3]`           | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
