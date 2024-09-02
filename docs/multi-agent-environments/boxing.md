
# Boxing

```{figure} ../_static/videos/multi-agent-environments/boxing.gif
:width: 140px
:name: boxing
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.atari import boxing_v2` |
|--------------------|------------------------------------------|
| Actions            | Discrete                                 |
| Parallel API       | Yes                                      |
| Manual Control     | No                                       |
| Agents             | `agents= ['first_0', 'second_0']`        |
| Agents             | 2                                        |
| Action Shape       | (1,)                                     |
| Action Values      | [0,17]                                   |
| Observation Shape  | (210, 160, 3)                            |
| Observation Values | (0,255)                                  |

*Boxing* is an adversarial game where precise control and appropriate responses to your opponent are key. The players have two minutes (around 1200 steps) to duke it out in the ring. Each step, they can move and punch. Successful punches score points, 1 point for a long jab, 2 for a close power punch, and 100 points for a KO (which also will end the game). Whenever you score a number of points, you are rewarded by that number and your opponent is penalized by that number.

[Official Boxing manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=45)

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

* v2: Minimal Action Space (1.18.0)
* v1: Breaking changes to entire API (1.4.0)
* v0: Initial versions release (1.0.0)
