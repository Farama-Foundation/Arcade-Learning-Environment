"""Registration for Atari environments."""

from __future__ import annotations

import ale_py.roms as roms
import gymnasium


def rom_id_to_name(rom: str) -> str:
    """Convert a rom id in snake_case to the name in PascalCase."""
    return rom.title().replace("_", "")


def register_v0_v4_envs():
    """Registers all v0 and v4 environments."""
    legacy_games = [
        "adventure",
        "air_raid",
        "alien",
        "amidar",
        "assault",
        "asterix",
        "asteroids",
        "atlantis",
        "bank_heist",
        "battle_zone",
        "beam_rider",
        "berzerk",
        "bowling",
        "boxing",
        "breakout",
        "carnival",
        "centipede",
        "chopper_command",
        "crazy_climber",
        "defender",
        "demon_attack",
        "double_dunk",
        "elevator_action",
        "enduro",
        "fishing_derby",
        "freeway",
        "frostbite",
        "gopher",
        "gravitar",
        "hero",
        "ice_hockey",
        "jamesbond",
        "journey_escape",
        "kangaroo",
        "krull",
        "kung_fu_master",
        "montezuma_revenge",
        "ms_pacman",
        "name_this_game",
        "phoenix",
        "pitfall",
        "pong",
        "pooyan",
        "private_eye",
        "qbert",
        "riverraid",
        "road_runner",
        "robotank",
        "seaquest",
        "skiing",
        "solaris",
        "space_invaders",
        "star_gunner",
        "tennis",
        "time_pilot",
        "tutankham",
        "up_n_down",
        "venture",
        "video_pinball",
        "wizard_of_wor",
        "yars_revenge",
        "zaxxon",
    ]

    for rom in legacy_games:
        for suffix, frameskip in (("", (2, 5)), ("NoFrameskip", 1)):
            for version, repeat_action_probability in (("v0", 0.25), ("v4", 0.0)):
                name = rom_id_to_name(rom)

                # Register the environment
                gymnasium.register(
                    id=f"{name}{suffix}-{version}",
                    entry_point="ale_py.env:AtariEnv",
                    kwargs=dict(
                        game=rom,
                        repeat_action_probability=repeat_action_probability,
                        full_action_space=False,
                        frameskip=frameskip,
                        max_num_frames_per_episode=108_000,
                    ),
                )


def register_v5_envs():
    """Register all v5 environments."""
    all_games = roms.get_all_rom_ids()

    for rom in all_games:
        # These roms don't have a single-player ROM attached (do have a multi-player mode)
        if rom in {"combat", "joust", "maze_craze", "warlords"}:
            continue

        name = rom_id_to_name(rom)

        # max_episode_steps is 108k frames which is 30 mins of gameplay.
        # This corresponds to 108k / 4 = 27,000 steps
        gymnasium.register(
            id=f"ALE/{name}-v5",
            entry_point="ale_py.env:AtariEnv",
            vector_entry_point="ale_py.vector_env:AtariVectorEnv",
            kwargs=dict(
                game=rom,
                repeat_action_probability=0.25,
                full_action_space=False,
                frameskip=4,
                max_num_frames_per_episode=108_000,
            ),
        )
