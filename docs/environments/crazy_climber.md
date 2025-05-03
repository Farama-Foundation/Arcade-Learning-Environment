---
title: CrazyClimber
---

# CrazyClimber

```{figure} ../_static/videos/environments/crazy_climber.gif
:width: 120px
:name: CrazyClimber
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                       |
|-------------------|---------------------------------------|
| Make              | gymnasium.make("ALE/CrazyClimber-v5") |
| Action Space      | Discrete(9)                           |
| Observation Space | Box(0, 255, (210, 160, 3), uint8)     |

For more CrazyClimber variants with different observation and action spaces, see the variants section.

## Description

You are a climber trying to reach the top of four buildings, while avoiding obstacles like closing windows and falling objects. When you receive damage (windows closing or objects) you will fall and lose one life; you have a total of 5 lives before the end games. At the top of each building, there's a helicopter which you need to catch to get to the next building. The goal is to climb as fast as possible while receiving the least amount of damage.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=113)

## Actions

CrazyClimber has the action space of `Discrete(9)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

|   Value | Meaning   |
|---------|-----------|
|       0 | NOOP      |
|       1 | UP        |
|       2 | RIGHT     |
|       3 | LEFT      |
|       4 | DOWN      |
|       5 | UPRIGHT   |
|       6 | UPLEFT    |
|       7 | DOWNRIGHT |
|       8 | DOWNLEFT  |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the q"rgb" type

See variants section for the type of observation used by each environment id by default.

### Reward

A table of scores awarded for completing each row of a building is provided on [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=113).

## Variants

CrazyClimber has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                     | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------------------|-------------|--------------|------------------------------|
| CrazyClimber-v0            | `rgb`       | `(2, 5)`     | `0.25`                       |
| CrazyClimberNoFrameskip-v0 | `rgb`       | `1`          | `0.25`                       |
| CrazyClimber-v4            | `rgb`       | `(2, 5)`     | `0.00`                       |
| CrazyClimberNoFrameskip-v4 | `rgb`       | `1`          | `0.00`                       |
| ALE/CrazyClimber-v5        | `rgb`       | `4`          | `0.25`                       |

See the [version history page](https://ale.farama.org/environments/#version-history-and-naming-schemes) to implement previously implemented environments, e.g., `CrazyClimberNoFrameskip-v4`.

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, 1, 2, 3]`    | `0`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
