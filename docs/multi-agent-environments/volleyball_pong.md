
# Volleyball Pong

```{figure} ../_static/videos/multi-agent-environments/volleyball_pong.gif
:width: 140px
:name: volleyball_pong
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import volleyball_pong_v2`        |
|----------------------|----------------------------------------------------------|
| Actions              | Discrete                                                 |
| Parallel API         | Yes                                                      |
| Manual Control       | No                                                       |
| Agents               | `agents= ['first_0', 'second_0', 'third_0', 'fourth_0']` |
| Agents               | 4                                                        |
| Action Shape         | (1,)                                                     |
| Action Values        | [0,5]                                                    |
| Observation Shape    | (210, 160, 3)                                            |
| Observation Values   | (0,255)                                                  |

Four player team battle. Get the ball onto your opponent's floor to score. In addition to being able to move left and right, each player can also jump higher to affect the ball's motion above the net. This is a team game, so a given team must try to coordinate to get the ball away from their scoring areas towards their opponent's. Specifically `first_0` and `third_0` are on one team and `second_0` and `fourth_0` are on the other. Scoring a point gives your team +1 reward and your opponent team -1 reward.

Serves are timed: If the player does not serve within 2 seconds of receiving the ball, their team receives -1 points, and the timer resets. This prevents one player from indefinitely stalling the game, but also means it is no longer a purely zero-sum game.

[Official Video Olympics manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)

## Environment parameters

Some environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments)..

Parameters specific to Volleyball Pong are

``` python
volleyball_pong_v3.env(num_players=4)
```

`num_players`:  Number of players (must be either 2 or 4)

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
