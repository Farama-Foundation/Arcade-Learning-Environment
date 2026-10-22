# FAQ

## What's the difference between the Atari environments in [OpenAI Gym](https://github.com/openai/gym) / [Gymnasium](https://github.com/farama-Foundation/gymnasium) and ALE?

The Atari environments provided in Gym and Gymnasium are built on the ALE; they just provide a different API to it. Since ALE 0.7, Gym uses `ale-py`, and since ALE 0.9 the Gymnasium environments are registered by `ale-py` itself, so there is no difference between using `ale-py` and Gymnasium.

## What is [Xitari](https://github.com/deepmind/xitari)? Should I use it?

Xitari is a fork of the ALE around version 0.4. It added support for more games and some other minor changes. This work has now been upstreamed to the ALE and it's recommended you use the ALE directly.

## I want to be able to extract from the game the number of lives my agent still has. How can I do it?

Since version 0.5.0, the ALE supports this through the `lives()` function. We strongly encourage RL researchers not to use the loss-of-life signal for episode termination and, if they do, to clearly report the setting used.

## When extracting the screen I realized that I never see a pixel with an odd value. However, the pixel is represented as a byte. Shouldn't it go up to 255 with odd and even values?

No, the Atari 2600 console (NTSC format) only supports 128 colours. Therefore, even though the colours are represented in a byte, only the seven most significant bits are used. Consequently, you have
to right-shift the colour byte by 1 bit to produce consecutively numbered colour values.

## Can I extract other state information such as the x,y position of sprites?

No, but there is a project which allows you to do just that: [AtariARI](https://github.com/mila-iqia/atari-representation-learning).
