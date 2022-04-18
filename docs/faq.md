# FAQ

### What's the difference between the Atari environments in [OpenAI Gym](https://github.com/openai/gym) and the ALE?

The environments provided in Gym are built on the ALE. It just provides a different API to the ALE. As of ALE 0.7 OpenAI Gym now uses `ale-py` and so there's no difference between using `ale-py` and Gym.

### What is [Xitari](https://github.com/deepmind/xitari)? Should I use it?

Xitari is a fork of the ALE around version 0.4. It added support for more games and some other minor changes. This work has now been upstreamed to the ALE and it's recommended you use the ALE directly.


### I downloaded ALE and I installed it successfully but I cannot find any ROM file at `roms/`. Do I have to get them somewhere else?

Yes. We do not distribute Atari 2600 ROMs. For the best compatability we recommend matching the MD5 sum of downloaded ROMs to those we support in the ALE. For a list of supported games checkout [`Arcade-Learning-Environment/md5.txt`](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/md5.txt).

If you're using the Python interface you can use the command line utility `ale-import-roms` to import supported ROMs.

### I want to be able to extract from the game the number of lives my agent still has. How can I do it?

Previous versions of ALE did not support this. We started to support such feature since version
0.5.0, through the use of the function `lives()`. We strongly encourage RL researchers not utilize the loss of life signal for episode termination, and if you do to clearly report the setting used.


### When extracting the screen I realized that I never see a pixel with an odd value. However, the pixel is represented as a byte. Shouldn't it be up to 255 with odd an even values?

No, the Atari 2600 console (NTSC format) only supports 128 colours. Therefore, even though the colours are represented  in a byte, in fact only the seven most significant bits are used. Consequently you have
to right-shift the colour byte by 1 bit to produce consecutively numbered colour values.


### Can I extract other state information such as the x,y position of sprites?

No, but there is a project which allows you to do just that: [AtariARI](https://github.com/mila-iqia/atari-representation-learning).
