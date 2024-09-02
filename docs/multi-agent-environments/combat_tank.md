
# Combat: Tank

```{figure} ../_static/videos/multi-agent-environments/combat_tank.gif
:width: 140px
:name: combat_tank
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.atari import combat_tank_v3`     |
|--------------------|---------------------------------------------------|
| Actions            | Discrete                                          |
| Parallel API       | Yes                                               |
| Manual Control     | No                                                |
| Agents             | `agents= ['first_0', 'second_0']`                 |
| Agents             | 2                                                 |
| Action Shape       | (1,)                                              |
| Action Values      | [0,5]                                             |
| Observation Shape  | (210, 160, 3)                                     |
| Observation Values | (0,255)                                           |

*Combat*'s classic tank mode is an adversarial game where prediction, and positioning are key. The players move around the map. When your opponent is hit by your bullet, you score a point. Note that your opponent gets blasted through obstacles when it is hit, potentially putting it in a good position to hit you back. Whenever you score a point, you are rewarded +1 and your opponent is penalized -1.

[Official Combat manual](https://atariage.com/manual_html_page.php?SoftwareID=935)

## Environment parameters

Some environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

Parameters specific to combat-tank are

``` python
combat_tank_v2.env(has_maze=True, is_invisible=False, billiard_hit=True)
```

* `has_maze`:  Set to true to have the map be a maze instead of an open field
* `is_invisible`:  If true, tanks are invisible unless they are firing or are running into a wall.
* `billiard_hit`:  If true, bullets bounce off walls, in fact, like billiards, they only count if they hit the opponent's tank after bouncing off a wall.

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

* v2: Minimal Action Space (1.18.0)
* v1: Breaking changes to entire API (1.4.0)
* v0: Initial versions release (1.0.0)
