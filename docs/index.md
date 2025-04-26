---
hide-toc: true
firstpage:
lastpage:
---

# Arcade Learning Environment

The Arcade Learning Environment (ALE), commonly referred to as Atari, is a framework that allows researchers and hobbyists to develop AI agents for Atari 2600 roms. Its built on top of the Atari 2600 emulator [Stella](https://github.com/stella-emu/stella) and separates the details of emulation from agent design. Users can interact with the games through the [Gymnasium API](gymnasium-interface), [Python interface](python-interface) and [C++ interface](cpp-interface). For an overview of our goals for the ALE read [The Arcade Learning Environment: An Evaluation Platform for General Agents](https://jair.org/index.php/jair/article/view/10819) and if you use ALE in your research, we ask that you please [cite](./citing.md) the appropriate paper(s) in reference to the environment.

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

# Initialise the environment
env = gym.make("ALE/Breakout-v5", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

```{toctree}
:hidden:
:caption: Introduction

getting-started
env-spec
environments
vector-environment
multi-agent-environments
faq
citing
```

```{toctree}
:hidden:
:caption: API

gymnasium-interface
cpp-interface
python-interface
visualization
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Arcade-Learning-Environment>
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/docs/README.md>
```
