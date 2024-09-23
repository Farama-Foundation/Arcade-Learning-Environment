---
title: Phoenix
---

# Phoenix

```{figure} ../../_static/videos/environments/phoenix.gif
:width: 120px
:name: Phoenix
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                   |
|-------------------|-----------------------------------|
| Action Space      | Discrete(8)                       |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Creation          | make(ALE/Phoenix-v5)              |

For more Phoenix variants with different observation and action spaces, see the variants section.

## Description

Your goal is to reach and shoot the alien pilot. On your way there, you must eliminate waves of war birds while avoiding their bombs.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=355)

## Actions

Phoenix has the action space of `Discrete(8)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

| Value   | Meaning    | Value   | Meaning    | Value   | Meaning     |
|---------|------------|---------|------------|---------|-------------|
| `0`     | `NOOP`     | `1`     | `FIRE`     | `2`     | `RIGHT`     |
| `3`     | `LEFT`     | `4`     | `DOWN`     | `5`     | `RIGHTFIRE` |
| `6`     | `LEFTFIRE` | `7`     | `DOWNFIRE` |         |             |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.


## Variants

Phoenix has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id         | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------|-------------|--------------|------------------------------|
| ALE/Phoenix-v5 | `"rgb"`     | `1`          | `0.00`                       |

See the [version history page](https://ale.farama.org/environments/#version-history-and-naming-schemes) to implement previously implemented environments, e.g., `PhoenixNoFrameskip-v4`.

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0]`             | `0`            | `[0]`                    | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
