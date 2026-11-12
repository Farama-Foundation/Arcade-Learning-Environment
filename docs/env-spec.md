# Environment Specifications

This section provides additional information regarding the environments implemented in the ALE.

## Available Actions

The following regular actions are defined by the `Action` enum in [`common/Constants.h`](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/common/Constants.h). These can also be accessed in Python through the enum object `ale_py.Action`. These actions are interpreted by ALE as follows:

| Index | Action                  | Description                                                                     |
|-------|-------------------------|---------------------------------------------------------------------------------|
| 0     | **`NOOP`**              | No operation, do nothing.                                                       |
| 1     | **`FIRE`**              | Press the fire button without updating the joystick position.                   |
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

<small>1</small>: Note that the **`RESET`** action toggles the Atari 2600 reset switch, rather than resetting the
environment, and as such is ignored by most interfaces.

Note: There are two main types of controllers on the Atari 2600 console: the [joystick controller](https://en.wikipedia.org/wiki/Atari_CX40_joystick) and the [paddle controller](https://en.wikipedia.org/wiki/Paddle_\(game_controller\)). For paddle controllers all **`*RIGHT*`** actions correspond to a Δ-movement to the right on the wheel, and all **`*LEFT*`** actions correspond to a Δ-movement to the left.


## Terminal States

Once the end of an episode is reached (a terminal state in RL terminology), no further emulation
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
actual choice. This value can be modified using the option `repeat_action_probability`.
The default value was chosen as the highest value for which human play-testers
were unable to detect any delay or control lag. ([Machado et al. 2018](#ref-1)).

The motivation for introducing action repeat stochasticity was to help separate _trajectory optimization_ research from _robust controller optimization_, the latter often being the
desired outcome in reinforcement learning (RL). We strongly encourage RL researchers to use
the default stochasticity level in their agents, and clearly report the setting used.

## Minimal Action Set

It may sometimes be convenient to restrict the agent to a smaller action set. This can be
accomplished with the `ALEInterface` method
`getMinimalActionSet`, which returns a set of actions judged "minimal" to play a given
game. Due to the potentially high impact of this setting on performance, we encourage researchers
to clearly report the method used in their experiments.


## Modes and Difficulties

ALE 0.6.0 introduced modes and difficulties, which can be set using the methods `setMode` and `setDifficulty`. These introduce a whole range of new environments. For more details, see [Machado et al. 2018](#ref-1).

## Settings

The C++ and Python interfaces are configured with `setBool`, `setInt`, `setFloat` and `setString` (with matching getters). Most settings are read when a ROM is loaded, so set them **before** calling `loadROM`.

| Key                          | Type    | Default        | Description                                                                                                                     |
|------------------------------|---------|----------------|---------------------------------------------------------------------------------------------------------------------------------|
| `random_seed`                | `int`   | `-1`           | Seed for the environment's random number generator (sticky actions, etc.). `-1` seeds from the current time.                     |
| `repeat_action_probability`  | `float` | `0.25`         | Probability of repeating the previous action instead of the chosen one (sticky actions), see above.                             |
| `frame_skip`                 | `int`   | `1`            | Number of frames each call to `act` emulates with the same action; the rewards are summed.                                      |
| `max_num_frames_per_episode` | `int`   | `0`            | Truncate the episode after this many frames; `0` disables the limit.                                                            |
| `truncate_on_loss_of_life`   | `bool`  | `false`        | Truncate the episode when a life is lost.                                                                                       |
| `reward_min`, `reward_max`   | `int`   | int min / max  | Clip the reward returned by `act` to this range.                                                                                |
| `color_averaging`            | `bool`  | `false`        | Average the colours of the current and previous frames to reduce flickering, see above.                                         |
| `paddle_min`, `paddle_max`   | `int`   | per game       | Override the range of paddle resistance values used by paddle games.                                                            |
| `sound_obs`                  | `bool`  | `false`        | Record the audio of each frame so that it can be read with `getAudio`.                                                          |
| `display_screen`             | `bool`  | `false`        | Display the screen in a window (requires SDL support).                                                                          |
| `sound`                      | `bool`  | `false`        | Play the game audio (requires SDL support).                                                                                     |
| `record_screen_dir`          | `str`   | `""`           | Directory to save every frame to as a PNG, see [visualization](visualization.md).                                               |
| `record_sound_filename`      | `str`   | `""`           | File to save the audio to as a WAV (requires `sound`), see [visualization](visualization.md).                                   |
| `system_random_seed`         | `int`   | `4753849`      | Seed for the emulator's internal randomness; fixed by default for full determinism.                                              |


## References


(ref-1)=
<a id="1">[1]</a>
Machado et al.
"Revisiting the Arcade Learning Environment: Evaluation Protocols
and Open Problems for General Agents"
Journal of Artificial Intelligence Research (2018)
URL: https://jair.org/index.php/jair/article/view/11182
