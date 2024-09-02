
# Double Dunk

```{figure} ../_static/videos/multi-agent-environments/double_dunk.gif
:width: 140px
:name: double_dunk
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.atari import double_dunk_v3` |
|--------------------|-----------------------------------------------|
| Actions            | Discrete                                      |
| Parallel API       | Yes                                           |
| Manual Control     | No                                            |
| Agents             | agents= ['first_0', 'second_0']               |
| Agents             | 2                                             |
| Action Shape       | (1,)                                          |
| Action Values      | [0,17]                                        |
| Observation Shape  | (210, 160, 3)                                 |
| Observation Values | (0,255)                                       |

An adversarial game that combines control and precise selection. The game has two stages: selection and play. Selection can be difficult because you have to hold the same action for a few steps and then take the 0 action. Strategy choice is timed: if a player does not select any action after 2 seconds (120 frames) then the player is rewarded -1, and the timer resets. This prevents one player from indefinitely stalling the game, but also means it is no longer a purely zero-sum game. Once play begins, each team has two players. You only control one at a time, and which one you control depends on the selected play. Scoring should be familiar to basketball fans (2-3 points per successful shot).

[Official double dunk manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=153)

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
* v2: No action timer (1.9.0)
* v1: Breaking changes to entire API (1.4.0)
* v0: Initial versions release (1.0.0)
