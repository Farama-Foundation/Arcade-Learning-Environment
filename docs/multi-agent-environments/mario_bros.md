
# Mario Bros

```{figure} ../_static/videos/multi-agent-environments/mario_bros.gif
:width: 140px
:name: mario_bros
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import mario_bros_v3` |
|----------------------|----------------------------------------------|
| Actions              | Discrete                                     |
| Parallel API         | Yes                                          |
| Manual Control       | No                                           |
| Agents               | `agents= ['first_0', 'second_0']`            |
| Agents               | 2                                            |
| Action Shape         | (1,)                                         |
| Action Values        | [0,17]                                       |
| Observation Shape    | (210, 160, 3)                                |
| Observation Values   | (0,255)                                      |

A mixed-sum game of planning and control. The main goal is to kick a pest off the floor. This requires 2 steps:

1. Hit the floor below the pest, flipping it over. This knocks the pest on its back.
2. You to move up onto the floor where the pest is and you can kick it off. This earns +800 reward

Note that since this process has two steps there are opportunities for the two agents to either collaborate by helping each other knock pests over and collect them (potentially allowing both to collect reward more quickly), or for agents to steal the other's work. If you run into an active pest or a fireball, you lose a life. If you lose all your lives, you are done, and the other player keeps playing. You can gain a new life after earning 20000 points. There are other ways of earning points, by collecting bonus coins or wafers, earning 800 points each.

[Official mario bros manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=286)

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
