
# Gymnasium Interface

ALE natively supports [Gymnasium](https://github.com/farama-Foundation/gymnasium). To use these new environments you can simply:

```py
import gymnasium as gym
import ale_py

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

## Continuous Action Space

By default, ALE supports discrete actions related to the cardinal directions and fire (e.g., `UP`, `DOWN`, `LEFT`, `FIRE`).
With `continuous`, Atari environment can be modified to support continuous actions, first proposed in [CALE: Continuous Arcade Learning Environment](https://arxiv.org/pdf/2410.23810).

To initialize an environment with continuous actions, simply use the argument `continuous=True` in the `gymnasium.make`:
```python
>>> import gymnasium as gym
>>> import numpy as np
>>> import ale_py

>>> gym.register_envs(ale_py)
>>> env = gym.make("ALE/Breakout-v5", continuous=True)
>>> env.action_space  # radius, theta and fire where radius and theta for polar coordinates
Box([0.0, -np.pi, 0.0], [1.0, np.pi, 1.0], np.float32)
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(np.array([0.9, 0.4, 0.7], dtype=np.float32))
```
