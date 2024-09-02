
# Joust

```{figure} ../_static/videos/multi-agent-environments/joust.gif
:width: 140px
:name: joust
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import joust_v3` |
|----------------------|-----------------------------------------|
| Actions              | Discrete                                |
| Parallel API         | Yes                                     |
| Manual Control       | No                                      |
| Agents               | `agents= ['first_0', 'second_0']`       |
| Agents               | 2                                       |
| Action Shape         | (1,)                                    |
| Action Values        | [0,17]                                  |
| Observation Shape    | (210, 160, 3)                           |
| Observation Values   | (0,255)                                 |

Mixed sum game involving scoring points in an unforgiving world. Careful positioning, timing, and control is essential, as well as awareness of your opponent.

In Joust, you score points by hitting the opponent and NPCs when you are above them. If you are below them, you lose a life. In a game, there are a variety of waves with different enemies and different point scoring systems. However, expect that you can earn around 3000 points per wave.

[Official joust manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=253)

## Environment parameters

Environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

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

* v3: Minimal Action Space (1.18.0)
* v2: Breaking changes to entire API (1.4.0)
* v1: Fixes to how all environments handle premature death (1.3.0)
* v0: Initial versions release (1.0.0)
