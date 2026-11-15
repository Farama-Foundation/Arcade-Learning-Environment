"""Record the learned-policy GIFs used by gen_environment_gifs.py.

These are inference-only, CPU recordings of MIT-licensed RL Zoo checkpoints:
https://github.com/DLR-RM/rl-trained-agents
No checkpoints are committed, and these packages are not docs-build dependencies.
The recordings were made with Python 3.12 and the following optional dependencies:

    uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
    uv pip install stable-baselines3==2.9.0 sb3-contrib==2.9.0 \
        gymnasium==1.4.0 ale-py==0.12.1 pillow==12.1.1 \
        opencv-python-headless==4.13.0.92 numpy==2.5.4

Run gen_environment_gifs.py with one or more ROM names (e.g. pong breakout).
Models are cached outside the repository, under $XDG_CACHE_HOME/ale-doc-gifs
(or ~/.cache/ale-doc-gifs). Only load checkpoints from this trusted source:
SB3 checkpoints contain pickled Python objects.

All games use seed 0, deterministic actions, no sticky actions, AtariWrapper
preprocessing (four-frame action repeat, 84x84 grayscale), and a four-frame
observation stack, matching the NoFrameskip-v4 training setup. The wrapper
restarts play and clears the observation stack after life loss, as in training;
evaluation continues until the underlying game's first full episode ends.
Rewards are not clipped. The window is the earliest 301 consecutive rendered
states within a single life with the highest sum of the 300 intervening step
rewards. There is no manual window selection and no life-reset splice in a GIF.
The full-episode return includes reset-time no-op/FIRE rewards and ends at game
over or the environment's time limit. Window rewards exclude reset-time rewards.

Frames are the unmodified RGB renderer output, not the policy's observations.
The main generator saves all 301 states using its standard 50 ms frame duration
and infinite loop. Pillow only merges consecutive states that are identical.
"""

import os
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image

REVISION = "cd35bde610f4045bf2e0731c8f4c88d22df8fc85"
POLICIES = {
    "pong": ("Pong", "qrdqn"),
    "breakout": ("Breakout", "qrdqn"),
    "space_invaders": ("SpaceInvaders", "qrdqn"),
    "qbert": ("Qbert", "qrdqn"),
    "seaquest": ("Seaquest", "qrdqn"),
    "ms_pacman": ("MsPacman", "dqn"),
    "beam_rider": ("BeamRider", "qrdqn"),
    "enduro": ("Enduro", "qrdqn"),
}
SEED = 0
WINDOW_STEPS = 300


def record_policy(game, algorithm, cache_dir):
    """Return a contiguous scoring window and unmodified episode returns."""
    import ale_py
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.save_util import load_from_zip_file

    gym.register_envs(ale_py)
    torch.set_num_threads(1)
    env_id = f"{game}NoFrameskip-v4"
    source = (
        f"https://raw.githubusercontent.com/DLR-RM/rl-trained-agents/{REVISION}/"
        f"{algorithm}/{env_id}_1/{env_id}.zip"
    )
    cache_dir = Path(cache_dir) / REVISION
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / f"{algorithm}-{env_id}.zip"
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix(".download")
        urlretrieve(source, temporary)
        temporary.replace(checkpoint)

    base = Monitor(
        gym.make(
            f"ALE/{game}-v5",
            render_mode="rgb_array",
            frameskip=1,
            repeat_action_probability=0.0,
        )
    )
    env = AtariWrapper(base, clip_reward=False)
    try:
        # Replace obsolete Gym spaces and training-only schedules in old archives.
        custom_objects = {
            "observation_space": gym.spaces.Box(0, 255, (4, 84, 84), np.uint8),
            "action_space": env.action_space,
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "exploration_schedule": lambda _: 0.0,
        }
        data, parameters, _ = load_from_zip_file(
            checkpoint, device="cpu", custom_objects=custom_objects
        )
        # Load only the policy, not obsolete optimizer states or replay buffers.
        policy = data["policy_class"](
            data["observation_space"],
            data["action_space"],
            lr_schedule=lambda _: 0.0,
            **data["policy_kwargs"],
        )
        policy.load_state_dict(parameters["policy"])
        policy.set_training_mode(False)
        del data, parameters
        observation, _ = env.reset(seed=SEED)
        # Same zero-initialized, channel-first stack as SB3's VecFrameStack.
        stack = np.zeros((4, 84, 84), dtype=np.uint8)
        stack[-1] = observation[..., 0]
        frames = deque([Image.fromarray(env.render())], maxlen=WINDOW_STEPS + 1)
        rewards = deque(maxlen=WINDOW_STEPS)
        best_frames = None
        best_score = float("-inf")
        best_start = 0
        step = 0
        while True:
            action, _ = policy.predict(stack, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            step += 1
            frames.append(Image.fromarray(env.render()))
            rewards.append(reward)
            if len(frames) == WINDOW_STEPS + 1:
                score = sum(rewards)
                if score > best_score:
                    best_frames = list(frames)
                    best_score = score
                    best_start = step - WINDOW_STEPS
            if base.get_episode_rewards():
                break
            if terminated or truncated:
                observation, _ = env.reset()
                if base.get_episode_rewards():
                    break
                stack.fill(0)
                frames.clear()
                frames.append(Image.fromarray(env.render()))
                rewards.clear()
            else:
                stack[:-1] = stack[1:]
            stack[-1] = observation[..., 0]
        if best_frames is None:
            raise ValueError(f"{game}: episode is shorter than the recording window")
        return best_frames, {
            "game": game,
            "algorithm": algorithm,
            "source": source,
            "revision": REVISION,
            "seed": SEED,
            "repeat_action_probability": 0.0,
            "window_start_step": best_start,
            "window_end_step": best_start + WINDOW_STEPS,
            "window_score": best_score,
            "episode_score": base.get_episode_rewards()[0],
            "episode_steps": step,
            "episode_ale_frames": base.get_episode_lengths()[0],
        }
    finally:
        env.close()


def generate_policy_frames(rom_name):
    """Download the pinned policy and return its recording frames and scores."""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return record_policy(*POLICIES[rom_name], cache_dir / "ale-doc-gifs")
