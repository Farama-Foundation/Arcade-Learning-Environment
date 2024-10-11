---
title: Krull
---

# Krull

```{figure} ../_static/videos/environments/krull.gif
:width: 120px
:name: Krull
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                   |
|-------------------|-----------------------------------|
| Action Space      | Discrete(18)                      |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |
| Import            | `gymnasium.make("ALE/Krull-v5")`  |

For more Krull variants with different observation and action spaces, see the variants section.

## Description

Your mission is to find and enter the Beast's Black Fortress, rescue Princess Lyssa, and destroy the Beast.The task is not an easy one, for the location of the Black Fortress changes with each sunrise on Krull.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=267)

## Actions

Krull has the action space `Discrete(18)` with the table below listing the meaning of each action's meanings.
As Krull uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`.

| Value   | Meaning      | Value   | Meaning         | Value   | Meaning        |
|---------|--------------|---------|-----------------|---------|----------------|
| `0`     | `NOOP`       | `1`     | `FIRE`          | `2`     | `UP`           |
| `3`     | `RIGHT`      | `4`     | `LEFT`          | `5`     | `DOWN`         |
| `6`     | `UPRIGHT`    | `7`     | `UPLEFT`        | `8`     | `DOWNRIGHT`    |
| `9`     | `DOWNLEFT`   | `10`    | `UPFIRE`        | `11`    | `RIGHTFIRE`    |
| `12`    | `LEFTFIRE`   | `13`    | `DOWNFIRE`      | `14`    | `UPRIGHTFIRE`  |
| `15`    | `UPLEFTFIRE` | `16`    | `DOWNRIGHTFIRE` | `17`    | `DOWNLEFTFIRE` |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the q"rgb" type

See variants section for the type of observation used by each environment id by default.

### Reward

You will receive various scores for each monster you kill. You can play the game until you have lost all your lives. For a more detailed documentation, consult [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=267).

## Variants

Krull has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                    | obs_type=   | frameskip=   | repeat_action_probability=   |
|---------------------------|-------------|--------------|------------------------------|
| Krull-v0                  | `"rgb"`     | `(2, 5)`     | `0.25`                       |
| Krull-ram-v0              | `"ram"`     | `(2, 5)`     | `0.25`                       |
| Krull-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       |
| Krull-ramNoFrameskip-v0   | `"ram"`     | `1`          | `0.25`                       |
| KrullDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       |
| KrullNoFrameskip-v0       | `"rgb"`     | `1`          | `0.25`                       |
| Krull-v4                  | `"rgb"`     | `(2, 5)`     | `0.0`                        |
| Krull-ram-v4              | `"ram"`     | `(2, 5)`     | `0.0`                        |
| Krull-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        |
| Krull-ramNoFrameskip-v4   | `"ram"`     | `1`          | `0.0`                        |
| KrullDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        |
| KrullNoFrameskip-v4       | `"rgb"`     | `1`          | `0.0`                        |
| ALE/Krull-v5              | `"rgb"`     | `4`          | `0.25`                       |
| ALE/Krull-ram-v5          | `"ram"`     | `4`          | `0.25`                       |

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