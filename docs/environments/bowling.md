---
title: Bowling
---

# Bowling

```{figure} ../_static/videos/environments/bowling.gif
:width: 120px
:name: Bowling
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                   |
|-------------------|-----------------------------------|
| Make              | gymnasium.make("ALE/Bowling-v5")  |
| Action Space      | Discrete(6)                       |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |

For more Bowling variants with different observation and action spaces, see the variants section.

## Description

Your goal is to score as many points as possible in the game of Bowling. A game consists of 10 frames and you have two tries per frame. Knocking down all pins on the first try is called a "strike". Knocking down all pins on the second roll is called a "spar". Otherwise, the frame is called "open".

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=879)

## Actions

Bowling has the action space of `Discrete(6)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

|   Value | Meaning   |
|---------|-----------|
|       0 | NOOP      |
|       1 | FIRE      |
|       2 | UP        |
|       3 | DOWN      |
|       4 | UPFIRE    |
|       5 | DOWNFIRE  |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the q"rgb" type

See variants section for the type of observation used by each environment id by default.

### Reward

You receive points for knocking down pins. The exact score depends on whether you manage a "strike", "spare" or "open" frame. Moreover, the points you score for one frame may depend on following frames. You can score up to 300 points in one game (if you manage to do 12 strikes). For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareID=879).

## Variants

Bowling has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id                | obs_type=   | frameskip=   | repeat_action_probability=   |
|-----------------------|-------------|--------------|------------------------------|
| Bowling-v0            | `rgb`       | `(2, 5)`     | `0.25`                       |
| BowlingNoFrameskip-v0 | `rgb`       | `1`          | `0.25`                       |
| Bowling-v4            | `rgb`       | `(2, 5)`     | `0.00`                       |
| BowlingNoFrameskip-v4 | `rgb`       | `1`          | `0.00`                       |
| ALE/Bowling-v5        | `rgb`       | `4`          | `0.25`                       |

See the [version history page](https://ale.farama.org/environments/#version-history-and-naming-schemes) to implement previously implemented environments, e.g., `BowlingNoFrameskip-v4`.

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, 2, 4]`       | `0`            | `[0, 1]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
