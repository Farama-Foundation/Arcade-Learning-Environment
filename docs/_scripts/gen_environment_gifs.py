import os

import ale_py
import gymnasium as gym
from PIL import Image

gym.register_envs(ale_py)

# how many steps to record an env for
LENGTH = 300


if __name__ == "__main__":
    for rom_name in ale_py.roms.get_all_rom_ids():
        env_name = ale_py.registration._rom_id_to_name(rom_name)
        env = gym.make(f"ALE/{rom_name}-v5", render_mode="rgb_array")

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

        # make sure video doesn't already exist
        # if not os.path.exists(os.path.join(v_path, env_name + ".gif")):

        # render_fps = env.metadata.get("render_fps", 30)
        video_path = os.path.join(
            "..", "_static", "videos", "environments", f"{rom_name}.gif"
        )
        frames[0].save(
            video_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # milliseconds for the frame
            loop=0,
        )
