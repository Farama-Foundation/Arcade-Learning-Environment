
# Combat: Plane

```{figure} ../_static/videos/multi-agent-environments/combat_plane.gif
:width: 140px
:name: combat_plane
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.atari import combat_jet_v1` |
|--------------------|----------------------------------------------|
| Actions            | Discrete                                     |
| Parallel API       | Yes                                          |
| Manual Control     | No                                           |
| Agents             | `agents= ['first_0', 'second_0']`            |
| Agents             | 2                                            |
| Action Shape       | (1,)                                         |
| Action Values      | [0,17]                                       |
| Observation Shape  | (256, 160, 3)                                |
| Observation Values | (0,255)                                      |

*Combat*'s plane mode is an adversarial game where timing, positioning, and keeping track of your opponent's complex movements are key. The players fly around the map, able to control flight direction but not your speed. When your opponent is hit by your bullet, you score a point. Whenever you score a point, you are rewarded +1 and your opponent is penalized -1.

[Official Combat manual](https://atariage.com/manual_html_page.php?SoftwareID=935)

## Environment parameters

Some environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

Parameters specific to combat-plane are

``` python
combat_plane_v2.env(game_version="jet", guided_missile=True)
```

* `game_version`:  Accepted arguments are "jet" or "bi-plane". Whether the plane is a bi-plane or a jet. (Jets move faster)
* `guided_missile`:  Whether the missile can be directed after being fired, or whether it is on a fixed path.

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
