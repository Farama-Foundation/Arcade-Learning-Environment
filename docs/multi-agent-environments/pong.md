
# Pong

```{figure} ../_static/videos/multi-agent-environments/pong.gif
:width: 140px
:name: pong
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import pong_v3` |
|----------------------|----------------------------------------|
| Actions              | Discrete                               |
| Parallel API         | Yes                                    |
| Manual Control       | No                                     |
| Agents               | `agents= ['first_0', 'second_0']`      |
| Agents               | 2                                      |
| Action Shape         | (1,)                                   |
| Action Values        | [0,5]                                  |
| Observation Shape    | (210, 160, 3)                          |
| Observation Values   | (0,255)                                |

Classic two player competitive game of timing. Get the ball past the opponent. Scoring a point gives you +1 reward and your opponent -1 reward. Serves are timed: If the player does not serve within 2 seconds of receiving the ball, they receive -1 points, and the timer resets. This prevents one player from indefinitely stalling the game, but also means it is no longer a purely zero-sum game.

[Official Video Olympics manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)

## Environment parameters

Some environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

Parameters specific to Pong are

``` python
pong_v3.env(num_players=2)
```

* `num_players`:  Number of players (must be either 2 or 4)

## Action Space (Minimal)

In any given turn, an agent can choose from one of 6 actions.

| Value   | Meaning      | Value   | Meaning         | Value   | Meaning        |
|---------|--------------|---------|-----------------|---------|----------------|
| `0`     | `NOOP`       | `1`     | `FIRE`          | `2`     | `UP`           |
| `3`     | `RIGHT`      | `4`     | `LEFT`          | `5`     | `DOWN`         |

See [environment specification](../env-spec) to see more information on the action meaning.

## Version History

* v3: Minimal Action Space (1.18.0)
* v2: No action timer (1.9.0)
* v1: Breaking changes to entire API (1.4.0)
* v0: Initial versions release (1.0.0)
