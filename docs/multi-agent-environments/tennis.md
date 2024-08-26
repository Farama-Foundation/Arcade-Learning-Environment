
# Tennis

```{figure} ../_static/videos/multi-agent-environments/tennis.gif
:width: 140px
:name: tennis
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import tennis_v3` |
|----------------------|------------------------------------------|
| Actions              | Discrete                                 |
| Parallel API         | Yes                                      |
| Manual Control       | No                                       |
| Agents               | `agents= ['first_0', 'second_0']`        |
| Agents               | 2                                        |
| Action Shape         | (1,)                                     |
| Action Values        | [0,17]                                   |
| Observation Shape    | (210, 160, 3)                            |
| Observation Values   | (0,255)                                  |

A competitive game of positioning and prediction. Goal: Get the ball past your opponent. Don't let the ball get past you. When a point is scored (by the ball exiting the area), you get +1 reward and your opponent gets -1 reward. Unlike normal tennis matches, the number of games won is not directly rewarded.

Serves are timed: If the player does not serve within 3 seconds of receiving the ball, they receive -1 points, and the timer resets. This prevents one player from indefinitely stalling the game, but also means it is no longer a purely zero-sum game.

[Official tennis manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=555)

## Environment parameters

Environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

## Action Space

In any given turn, an agent can choose from one of 18 actions.

| Action | Behavior       |
|:------:|----------------|
|   0    | No operation   |
|   1    | Fire           |
|   2    | Move up        |
|   3    | Move right     |
|   4    | Move left      |
|   5    | Move down      |
|   6    | Move upright   |
|   7    | Move upleft    |
|   8    | Move downright |
|   9    | Move downleft  |
|   10   | Fire up        |
|   11   | Fire right     |
|   12   | Fire left      |
|   13   | Fire down      |
|   14   | Fire upright   |
|   15   | Fire upleft    |
|   16   | Fire downright |
|   17   | Fire downleft  |

## Version History

* v3: Minimal Action Space (1.18.0)
* v2: No action timer (1.9.0)
* v1: Breaking changes to entire API (1.4.0)
* v0: Initial versions release (1.0.0)