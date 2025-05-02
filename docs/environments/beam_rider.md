---
title: BeamRider
---

# BeamRider

```{figure} ../_static/videos/environments/beam_rider.gif
:width: 120px
:name: BeamRider
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                    |
|-------------------|------------------------------------|
| Make              | gymnasium.make("ALE/BeamRider-v5") |
| Action Space      | Discrete(9)                        |
| Observation Space | Box(0, 255, (210, 160, 3), uint8)  |

For more BeamRider variants with different observation and action spaces, see the variants section.

## Description

You control a space-ship that travels forward at a constant speed. You can only steer it sideways between discrete positions. Your goal is to destroy enemy ships, avoid their attacks and dodge space debris.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=860)

## Actions

BeamRider has the action space of `Discrete(9)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

|   Value | Meaning   |
|---------|-----------|
|       0 | NOOP      |
|       1 | FIRE      |
|       2 | UP        |
|       3 | RIGHT     |
|       4 | LEFT      |
|       5 | UPRIGHT   |
|       6 | UPLEFT    |
|       7 | RIGHTFIRE |
|       8 | LEFTFIRE  |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the q"rgb" type

See variants section for the type of observation used by each environment id by default.

### Reward

You score points for destroying enemies. For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SystemID=2600&SoftwareID=860&itemTypeID=MANUAL).

## Variants

BeamRider has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                  | obs_type=   | frameskip=   | repeat_action_probability=   |
|-------------------------|-------------|--------------|------------------------------|
| BeamRider-v0            | `rgb`       | `(2, 5)`     | `0.25`                       |
| BeamRiderNoFrameskip-v0 | `rgb`       | `1`          | `0.25`                       |
| BeamRider-v4            | `rgb`       | `(2, 5)`     | `0.00`                       |
| BeamRiderNoFrameskip-v4 | `rgb`       | `1`          | `0.00`                       |
| ALE/BeamRider-v5        | `rgb`       | `4`          | `0.25`                       |

See the [version history page](https://ale.farama.org/environments/#version-history-and-naming-schemes) to implement previously implemented environments, e.g., `BeamRiderNoFrameskip-v4`.

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0]`             | `0`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
