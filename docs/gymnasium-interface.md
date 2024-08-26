# Gymnasium Interface

ALE natively supports [Gymnasium](https://github.com/farama-Foundation/gymnasium). To use these new environments you can simply:

```py
import gymnasium as gym

env = gym.make('ALE/Breakout-v5')
```

or any of the other environment IDs (e.g., `SpaceInvaders, Breakout, Freeway`, etc.).

For the list of available environments, see the [environment](environments.md) page

## Visualization

Gymnasium supports the `.render()` method on environments that supports frame perfect visualization, proper scaling, and audio support. The `render_mode` argument supports either `human | rgb_array`. For example,

```py
import gymnasium as gym

env = gym.make('Breakout-v0', render_mode='human')
```
