from __future__ import annotations

import ale_py.roms as roms
import gymnasium


def rom_id_to_name(rom: str) -> str:
    """
    Let the ROM ID be the ROM identifier in snake_case.
        For example, `space_invaders`
    The ROM name is the ROM ID in pascalcase.
        For example, `SpaceInvaders`

    This function converts the ROM ID to the ROM name.
        i.e., snakecase -> pascalcase
    """
    return rom.title().replace("_", "")


def register_envs():
    all_games = roms.get_all_rom_ids()

    for game in all_games:
        gymnasium.register(
            id=f"ALE/{game}-v6",
            entry_point="ale_py.env:AtariEnv",
            kwargs={
                "game": game,
                "obs_type": "rgb",
                "frameskip": 1,
                # max_episode_steps is 108k frames which is 30 mins of gameplay.
                # This corresponds to 108k / 4 = 27,000 steps
                "max_num_frames_per_episode": 108_000,
                "repeat_action_probability": 0,
                "full_action_space": False,
            },
        )
