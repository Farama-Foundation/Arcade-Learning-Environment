# Atari 57

The "Atari 57" set contains the 57 games used by DeepMind to benchmark agents, from DQN [[1]](#ref1) through Rainbow, R2D2, MuZero and Agent57 [[2]](#ref2). Reported "median" or "mean human-normalized scores" for Atari usually refer to this set of games.

All 57 games are supported by the ALE. Note that `Defender` and `Surround` were historically unavailable in some ALE-based toolkits (e.g., `atari-py` used by early versions of OpenAI Gym), so some papers benchmark on only 55 of these games. For the games previously available in OpenAI Gym, see [Previously in OpenAI Gym](gym).

```{raw} html
:file: atari_57.html
```

## References

(ref1)=
<a id="ref1">[1]</a>
Mnih et al.
"Human-level control through deep reinforcement learning"
Nature (2015)
URL: https://www.nature.com/articles/nature14236

(ref2)=
<a id="ref2">[2]</a>
Badia et al.
"Agent57: Outperforming the Atari Human Benchmark"
International Conference on Machine Learning (2020)
URL: https://arxiv.org/abs/2003.13350
