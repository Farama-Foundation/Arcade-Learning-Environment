---
title: Amidar
---

# Amidar

```{figure} ../_static/videos/environments/amidar.gif
:width: 120px
:name: Amidar
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                   |
|-------------------|-----------------------------------|
| Make              | gymnasium.make("ALE/Amidar-v5")   |
| Action Space      | Discrete(10)                      |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |

For more Amidar variants with different observation and action spaces, see the variants section.

## Description

This game is similar to Pac-Man, you are trying to visit all places on a 2-dimensional grid while simultaneously avoiding your enemies. You can turn the tables at one point in the game with your enemies turn into chickens and you can catch them.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=817)

## Actions

Amidar has the action space of `Discrete(10)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

|   Value | Meaning   |
|---------|-----------|
|       0 | NOOP      |
|       1 | FIRE      |
|       2 | UP        |
|       3 | RIGHT     |
|       4 | LEFT      |
|       5 | DOWN      |
|       6 | UPFIRE    |
|       7 | RIGHTFIRE |
|       8 | LEFTFIRE  |
|       9 | DOWNFIRE  |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the q"rgb" type

See variants section for the type of observation used by each environment id by default.

### Reward

You score points by traversing new parts of the grid. Coloring an entire box in the maze or catching chickens gives extra points. For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=817).

## Variants

Amidar has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id               | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------------|-------------|--------------|------------------------------|
| Amidar-v0            | `rgb`       | `(2, 5)`     | `0.25`                       |
| AmidarNoFrameskip-v0 | `rgb`       | `1`          | `0.25`                       |
| Amidar-v4            | `rgb`       | `(2, 5)`     | `0.00`                       |
| AmidarNoFrameskip-v4 | `rgb`       | `1`          | `0.00`                       |
| ALE/Amidar-v5        | `rgb`       | `4`          | `0.25`                       |

See the [version history page](https://ale.farama.org/environments/#version-history-and-naming-schemes) to implement previously implemented environments, e.g., `AmidarNoFrameskip-v4`.

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0]`             | `0`            | `[0, 3]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
