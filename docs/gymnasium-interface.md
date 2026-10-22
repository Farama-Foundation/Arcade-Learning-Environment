# Gymnasium Interface

ALE natively supports [Gymnasium](https://github.com/farama-Foundation/gymnasium). To use these environments, register them with Gymnasium and create one by its ID:

```py
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5')
```

The same works for any of the other environment IDs (e.g., `ALE/SpaceInvaders-v5`, `ALE/Freeway-v5`).

For the list of available environments, see the [environments](environments.md) page.

## Visualization

Gymnasium supports the `.render()` method on environments, which provides frame-perfect visualization, proper scaling, and audio support. The `render_mode` argument supports either `"human"` or `"rgb_array"`. For example,

```py
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5', render_mode='human')
```

## Continuous Action Space

By default, ALE supports discrete actions related to the cardinal directions and fire (e.g., `UP`, `DOWN`, `LEFT`, `FIRE`).
With `continuous`, Atari environments can be modified to support continuous actions, first proposed in [CALE: Continuous Arcade Learning Environment](https://arxiv.org/pdf/2410.23810).

To initialize an environment with continuous actions, simply pass the argument `continuous=True` to `gymnasium.make`:
```python
>>> import gymnasium as gym
>>> import numpy as np
>>> import ale_py

>>> gym.register_envs(ale_py)
>>> env = gym.make("ALE/Breakout-v5", continuous=True)
>>> env.action_space  # radius, theta and fire, where radius and theta are polar coordinates
Box([0.0, -np.pi, 0.0], [1.0, np.pi, 1.0], np.float32)
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(np.array([0.9, 0.4, 0.7], dtype=np.float32))
```
