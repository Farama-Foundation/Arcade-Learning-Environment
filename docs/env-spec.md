# Environment Specifications

This section provides additional information regarding the environment implemented in ALE.

## Available Actions

The following regular actions are defined by the `Action` enum in [`common/Constants.h`](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/src/common/Constants.h). These can also be accessed in Python through the enum object `ale_py.Action`. These actions are interpreted by ALE as follows:

| Index | Action                  | Description                                                                     |
|-------|-------------------------|---------------------------------------------------------------------------------|
| 0     | **`NOOP`**              | No operation, do nothing.                                                       |
| 1     | **`FIRE`**              | Press the fire button without updating the joystick position                    |
| 2     | **`UP`**                | Apply a Δ-movement upwards on the joystick                                      |
| 3     | **`RIGHT`**             | Apply a Δ-movement rightward on the joystick                                    |
| 4     | **`LEFT`**              | Apply a Δ-movement leftward on the joystick                                     |
| 5     | **`DOWN`**              | Apply a Δ-movement downward on the joystick                                     |
| 6     | **`UPRIGHT`**           | Execute **`UP`** and **`RIGHT`**                                                |
| 7     | **`UPLEFT`**            | Execute **`UP`** and **`LEFT`**                                                 |
| 8     | **`DOWNRIGHT`**         | Execute **`DOWN`** and **`RIGHT`**                                              |
| 9     | **`DOWNLEFT`**          | Execute **`DOWN`** and **`LEFT`**                                               |
| 10    | **`UPFIRE`**            | Execute **`UP`** and **`FIRE`**                                                 |
| 11    | **`RIGHTFIRE`**         | Execute **`RIGHT`** and **`FIRE`**                                              |
| 12    | **`LEFTFIRE`**          | Execute **`LEFT`** and **`FIRE`**                                               |
| 13    | **`DOWNFIRE`**          | Execute **`DOWN`** and **`FIRE`**                                               |
| 14    | **`UPRIGHTFIRE`**       | Execute **`UP`** and **`RIGHT`** and **`FIRE`**                                 |
| 15    | **`UPLEFTFIRE`**        | Execute **`UP`** and **`LEFT`** and **`FIRE`**                                  |
| 16    | **`DOWNRIGHTFIRE`**     | Execute **`DOWN`** and **`RIGHT`** and **`FIRE`**                               |
| 17    | **`DOWNLEFTFIRE`**      | Execute **`DOWN`** and **`LEFT`** and **`FIRE`**                                |
| 40    | **`RESET`**<sup>1</sup> | Toggles the Atari 2600 reset switch, **not** used for resetting the environment |

<small>1</small>: Note that the **`RESET`** action toggles the Atari 2600 reset switch, rather than reset the
environment, and as such is ignored by most interfaces.

Note: There are two main types of controllers on the Atari 2600 console. The [joystick controller](https://en.wikipedia.org/wiki/Atari_CX40_joystick) and the [paddle controller](https://en.wikipedia.org/wiki/Paddle_\(game_controller\)). For paddle controllers all **`*RIGHT*`** actions correspond to a Δ-movment to the right on the wheel, and all **`*LEFT*`** actions correspond to a Δ-movement to the left.


##  Terminal States

Once the end of episode is reached (a terminal state in RL terminology), no further emulation
takes place until the appropriate reset command is sent. This command is distinct from the Atari
2600 reset. This "system reset" avoids odd situations where the player can reset the game
through button presses, or where the game normally resets itself after a number of frames. This
makes for a cleaner environment interface. The interfaces described here all provide a system reset command or method.

## Color Averaging

Many Atari 2600 games display objects on alternating frames (sometimes even less frequently).
This can be an issue for agents that do not consider the whole screen history.
By default, _color averaging_ is **not** enabled, that is, the environment output is the actual frame from the emulator.
This behaviour can be turned on using `setBool` with the `color_averaging` key.

## Action Repeat Stochasticity

Beginning with ALE 0.5.0, there is now an option (enabled by default) to add
_action repeat stochasticity_ to the environment. With probability 𝗉 (default: 𝗉 = 0.25),
the previously executed action is executed again during the next frame, ignoring the agent's
actual choice. This value can be modified using the option `action_repeat_probability`.
The default value was chosen as the highest value for which human play-testers
were unable to detect any delay or control lag. ([Machado et al. 2018](#1)).

The motivation for introducing action repeat stochasticity was to help separate _trajectory optimization_ research from _robust controller optimization_, the latter often being the
desired outcome in reinforcement learning (RL). We strongly encourage RL researchers to use
the default stochasticity level in their agents, and clearly report the setting used.

## Minimal Action Set

It may sometimes be convenient to restrict the agent to a smaller action set. This can be
accomplished by querying the `RomSettings` class using the method
`getMinimalActionSet`. This then returns a set of actions judged "minimal" to play a given
game. Due to the potentially high impact of this setting on performance, we encourage researchers
to clearly report the method used in their experiments.


## Modes and Difficulties

ALE 0.6.0 introduces modes and difficulties, which can be set using the relevant methods `setMode`, `setDifficulty`. These introduce a whole range of new environments. For more details, see [Machado et al. 2018](#1).


## References


(#1)=
<a id="1">[1]</a>
Machado et al.
"Revisiting the Arcade Learning Environment: Evaluation Protocols
and Open Problems for General Agents"
Journal of Artificial Intelligence Research (2018)
URL: https://jair.org/index.php/jair/article/view/11182
