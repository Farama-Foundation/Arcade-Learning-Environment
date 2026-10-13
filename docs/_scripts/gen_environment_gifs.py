"""Generate documentation GIFs; optionally pass individual ROM names.

For example: python docs/_scripts/gen_environment_gifs.py pong breakout
Learned-policy dependencies and recording rules are in gen_policy_gifs.py.
Other games retain the random-action recording path.
"""

import argparse
import json
from pathlib import Path

import ale_py
import gymnasium as gym
from gen_policy_gifs import POLICIES, generate_policy_frames
from PIL import Image

gym.register_envs(ale_py)

# how many steps to record an env for
LENGTH = 300


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roms", nargs="*", help="ROM names (default: all games)")
    args = parser.parse_args()
    video_dir = Path(__file__).resolve().parents[1] / "_static/videos/environments"
    for rom_name in args.roms or ale_py.roms.get_all_rom_ids():
        video_path = video_dir / f"{rom_name}.gif"
        if rom_name in POLICIES:
            frames, report = generate_policy_frames(rom_name)
            print(json.dumps(report), flush=True)
        else:
            env_name = ale_py.registration._rom_id_to_name(rom_name)
            env = gym.make(f"ALE/{env_name}-v5", render_mode="rgb_array")

            # obtain and save LENGTH frames worth of steps
            frames = []
            env.reset()
            while len(frames) <= LENGTH:
                frames.append(Image.fromarray(env.render()))

                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    env.reset()
            env.close()

        frames[0].save(
            video_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # milliseconds for the frame
            loop=0,
        )
