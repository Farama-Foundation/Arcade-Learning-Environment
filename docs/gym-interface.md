# OpenAI Gym Interface

The ALE now natively supports OpenAI Gym. If you create any of the legacy Atari environments Gym is actually instantiating the ALE in the background. Although you could continue using the legacy environments as is we recommend using the new `v5` environments which follow the methodology set out in the paper [Revisiting the Arcade Environment by Machado et al. 2018](https://jair.org/index.php/jair/article/view/11182). To use these new environments you can simply:

```py
import gym
import ale_py

env = gym.make('ALE/Breakout-v5')
```

or any of the other environment IDs (e.g., `SpaceInvaders, Breakout, Freeway`, etc.).

## Visualization

Gym supports the `.render()` method on environments. We highly discourage users from using `render()` and instead to pass the `render_mode` keyword argument when constructing your environment. By doing so you'll be able to use our SDL2 visualization which supports: frame perfect visualization, proper scaling, and audio support. For example,

```py
import gym

env = gym.make('Breakout-v0', render_mode='human')
```

The `render_mode` argument supports either `human | rgb_array`. If `rgb_array` is specified we'll return the full RGB observation in the metadata dictionary returned after an agent step. For example,

```py
import gym

env = gym.make('Breakout-v0', render_mode='rgb_array')
env.reset()
_, _, _, metadata = env.step(0)
assert 'rgb_array' in metadata
```
