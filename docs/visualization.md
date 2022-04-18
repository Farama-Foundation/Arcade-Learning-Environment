# Visualization

ALE offers screen display and audio capabilities via the Simple DirectMedia Layer (SDL).
Instructions on how to install the SDL library, as well as enabling SDL support within ALE can be found in the section [Getting Started](./getting-started.md).
Screen display can be enabled using the boolean option `display_screen` (default: `false`),
and sound playback using the boolean option `sound` (default: `false`).

##  Recording Movies

ALE now provides support for recording frames; if sound is enabled, it is also possible to record audio output.
An example Python program is provided which will record both visual and audio output for a single episode of play. A similar example using C++ can be found at [`examples/video-recording`](https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/examples/video-recording).


### Python Example
```py
import os
import sys
from random import randrange
from ale_py import ALEInterface

def main(rom_file, record_dir):
    ale = ALEInterface()
    ale.setInt('random_seed', 123)

    # Enable screen display and sound output
    ale.setBool('display_screen', True)
    ale.setBool('sound', True)

    # Specify the recording directory and the audio file path
    ale.setString("record_screen_dir", record_dir) # Set the record directory
    ale.setString("record_sound_filename",
                    os.path.join(record_dir, "sound.wav"))

    ale.loadROM(rom_file)

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()
    num_actions = len(legal_actions)

    while not ale.game_over():
      a = legal_actions[randrange(num_actions)]
      ale.act(a)

    print(f"Finished episode. Frames can be found in {record_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
      print(f"Usage: {sys.argv[0]} rom_file record_dir")
      sys.exit()

    rom_file = sys.argv[1]
    record_dir = sys.argv[2]
    main(rom_file)
```

Once frames and/or sound have been recorded, they may be joined into a video using an external program like [ffmpeg](https://www.ffmpeg.org). For example, you can run:

```bash
# -r frame_rate
# -i input
# -f format
# -c:a audio_codec
# -c:v video_codec

ffmpeg -r 60 \
       -i record/%06d.png \
       -i record/sound.wav \
       -f mov \
       -c:a mp3 \
       -c:v libx264 \
       agent.mov
```

The parameters may vary depending on the format, you can find more examples at [`examples/video-recording`](https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/examples/video-recording).
