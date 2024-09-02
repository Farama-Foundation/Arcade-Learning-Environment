
# Video Checkers

```{figure} ../_static/videos/multi-agent-environments/video_checkers.gif
:width: 140px
:name: video_checkers
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import video_checkers_v4` |
|----------------------|--------------------------------------------------|
| Actions              | Discrete                                         |
| Parallel API         | Yes                                              |
| Manual Control       | No                                               |
| Agents               | `agents= ['first_0', 'second_0']`                |
| Agents               | 2                                                |
| Action Shape         | (1,)                                             |
| Action Values        | [0,4]                                            |
| Observation Shape    | (210, 160, 3)                                    |
| Observation Values   | (0,255)                                          |

A classical strategy game with arcade style controls.

Capture all of your opponents pieces by jumping over them. To move a piece, you must select a piece by hovering the cursor and pressing fire (action 1), moving the cursor, and pressing fire again. Note that the buttons must be held for multiple frames to be registered. If you win by capturing all your opponent's pieces, you are rewarded +1 and your opponent -1.

This is a timed game: if a player does not take a turn after 10 seconds, then that player is rewarded -1 points, their opponent is rewarded nothing, and the timer resets. This prevents one player from indefinitely stalling the game, but also means it is no longer a purely zero-sum game.

[Official video checkers manual](https://atariage.com/manual_html_page.php?SoftwareID=1427)

## Environment parameters

Environment parameters are common to all Atari environments and are described in the [base multi-agent environment documentation](../multi-agent-environments).

## Action Space (Minimal)

In any given turn, an agent can choose from one of 5 actions.

| Value | Meaning | Value | Meaning | Value | Meaning |
|-------|---------|-------|---------|-------|---------|
| `0`   | `NOOP`  | `1`   | `FIRE`  | `2`   | `UP`    |
| `3`   | `RIGHT` | `4`   | `LEFT`  |       |         |

See [environment specification](../env-spec) to see more information on the action meaning.

## Version History

* v4: Minimal Action Space (1.18.0)
* v3: No action timer (1.9.0)
* v2: Fixed checkers rewards (1.5.0)
* v1: Breaking changes to entire API (1.4.0)
* v0: Initial versions release (1.0.0)
