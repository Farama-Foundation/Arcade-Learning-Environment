
# Surround

```{figure} ../_static/videos/multi-agent-environments/surround.gif
:width: 140px
:name: surround
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import surround_v2` |
|----------------------|--------------------------------------------|
| Actions              | Discrete                                   |
| Parallel API         | Yes                                        |
| Manual Control       | No                                         |
| Agents               | `agents= ['first_0', 'second_0']`          |
| Agents               | 2                                          |
| Action Shape         | (1,)                                       |
| Action Values        | [0,4]                                      |
| Observation Shape    | (210, 160, 3)                              |
| Observation Values   | (0,255)                                    |

A competitive game of planning and strategy. In surround, your goal is to avoid the walls. If you run into a wall, you are rewarded -1 points, and your opponent, +1 points. But both players leave a trail of walls behind you, slowly filling the screen with obstacles. To avoid the obstacles as long as possible, you must plan your path to conserve space. Once that is mastered, a higher level aspect of the game comes into play, where both players literally try to surround the other with walls, so their opponent will run out of room and be forced to run into a wall.

[Official surround manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=943)

## Environment parameters

Environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

## Action Space (Minimal)

In any given turn, an agent can choose from one of 6 actions. (Fire is dummy action, but for the continuous numbering)

| Value   | Meaning      | Value   | Meaning         | Value   | Meaning        |
|---------|--------------|---------|-----------------|---------|----------------|
| `0`     | `NOOP`       | `1`     | `FIRE`          | `2`     | `UP`           |
| `3`     | `RIGHT`      | `4`     | `LEFT`          | `5`     | `DOWN`         |

See [environment specification](../env-spec) to see more information on the action meaning.

## Version History

* v2: Minimal Action Space (1.18.0)
* v1: Breaking changes to entire API (1.4.0)
* v0: Initial versions release (1.0.0)
