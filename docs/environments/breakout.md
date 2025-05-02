---
title: Breakout
---

# Breakout

```{figure} ../_static/videos/environments/breakout.gif
:width: 120px
:name: Breakout
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                   |
|-------------------|-----------------------------------|
| Make              | gymnasium.make("ALE/Breakout-v5") |
| Action Space      | Discrete(4)                       |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |

For more Breakout variants with different observation and action spaces, see the variants section.

## Description

You move a paddle and hit the ball in a brick wall at the top of the screen. Your goal is to destroy the brick wall. You can try to break through the wall and let the ball wreak havoc on the other side, all on its own! You have five lives.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=889)

## Actions

Breakout has the action space of `Discrete(4)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

|   Value | Meaning   |
|---------|-----------|
|       0 | NOOP      |
|       1 | FIRE      |
|       2 | RIGHT     |
|       3 | LEFT      |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the q"rgb" type

See variants section for the type of observation used by each environment id by default.

### Reward

You score points by destroying bricks in the wall. The reward for destroying a brick depends on the color of the brick. For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=889).

## Variants

Breakout has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                 | obs_type=   | frameskip=   | repeat_action_probability=   |
|------------------------|-------------|--------------|------------------------------|
| Breakout-v0            | `rgb`       | `(2, 5)`     | `0.25`                       |
| BreakoutNoFrameskip-v0 | `rgb`       | `1`          | `0.25`                       |
| Breakout-v4            | `rgb`       | `(2, 5)`     | `0.00`                       |
| BreakoutNoFrameskip-v4 | `rgb`       | `1`          | `0.00`                       |
| ALE/Breakout-v5        | `rgb`       | `4`          | `0.25`                       |

See the [version history page](https://ale.farama.org/environments/#version-history-and-naming-schemes) to implement previously implemented environments, e.g., `BreakoutNoFrameskip-v4`.

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes                                 | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------------------------------------|----------------|--------------------------|----------------------|
| `[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]` | `0`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
