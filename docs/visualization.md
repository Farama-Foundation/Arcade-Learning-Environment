# Visualization

ALE offers screen display and audio capabilities via the Simple DirectMedia Layer (SDL). Screen display can be enabled using the boolean option `display_screen` (default: `false`), and sound playback using the boolean option `sound` (default: `false`).

## Gymnasium API

[Gymnasium](https://github.com/farama-Foundation/gymnasium) provides two methods for visualizing an environment, human rendering and video recording.

### Human visualization

Through specifying the environment `render_mode="human"` then ALE will automatically create a window running at 60 frames per second showing the environment behaviour. It is highly recommended to close the environment after it has been used such that the rendering information is correctly shut down.

```python
import gymnasium
import ale_py

gymnasium.register_envs(ale_py)

env = gymnasium.make("ALE/Pong-v5", render_mode="human")
env.reset()
for _ in range(100):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Recording videos

Specifying the `render_mode="rgb_array"` will return the rgb array from `env.render()`, this can be combined with the `gymnasium.wrappers.RecordVideo` where the environment renders are stored and saved as mp4 videos for episodes.

The example below will record episodes on every other episode (`num % 2 == 0`) using the `episode_trigger` and save the folders in `saved-video-folder` with filename starting `video-` followed by the video number.

```python
import gymnasium
import ale_py

gymnasium.register_envs(ale_py)

env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array")
env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 2 == 0,
    video_folder="saved-video-folder",
    name_prefix="video-",
)
for episode in range(10):
    obs, info = env.reset()
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

env.close()
```

## Python Interface

ALE now provides support for recording frames; if sound is enabled, it is also possible to record audio output. An example Python program is provided which will record both visual and audio output for a single episode of play.

```python
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
    main(rom_file, "videos/")
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
