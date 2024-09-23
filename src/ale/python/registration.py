"""Registration for Atari environments."""

from __future__ import annotations

import ale_py.roms as roms
import gymnasium


def rom_id_to_name(rom: str) -> str:
    """Convert a rom id in snake_case to the name in PascalCase."""
    return rom.title().replace("_", "")


def register_envs():
    """Register all the Atari Environments."""
    all_rom_ids = roms.get_all_rom_ids()

    for rom_id in all_rom_ids:
        gymnasium.register(
            id=f"ALE/{rom_id_to_name(rom_id)}-v5",
            entry_point="ale_py.env:AtariEnv",
            kwargs={
                "game": rom_id,
                "obs_type": "rgb",
                "frameskip": 1,
                # max_episode_steps is 108k frames which is 30 mins of gameplay.
                # This corresponds to 108k / 4 = 27,000 steps
                "max_num_frames_per_episode": 108_000,
                "repeat_action_probability": 0,
                "full_action_space": False,
            },
        )
