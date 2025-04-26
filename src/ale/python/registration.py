"""Registration for Atari environments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Callable, NamedTuple

import ale_py.roms as roms
import gymnasium


class EnvFlavour(NamedTuple):
    """Environment flavour for env id suffix and kwargs."""

    suffix: str
    kwargs: Mapping[str, Any] | Callable[[str], Mapping[str, Any]]


class EnvConfig(NamedTuple):
    """Environment config for version, kwargs and flavours."""

    version: str
    kwargs: Mapping[str, Any]
    flavours: Sequence[EnvFlavour]


def _rom_id_to_name(rom: str) -> str:
    """Converts the Rom ID (snake_case) to ROM name in PascalCase.

    For example, `space_invaders` to `SpaceInvaders`
    """
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
    obs_types = ["rgb", "ram"]
    frameskip = defaultdict(lambda: 4, [("space_invaders", 3)])

    versions = [
        EnvConfig(
            version="v0",
            kwargs={
                "repeat_action_probability": 0.25,
                "full_action_space": False,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[
                # Default for v0 has 10k steps, no idea why...
                EnvFlavour("", {"frameskip": (2, 5)}),
                # Deterministic has 100k steps, close to the standard of 108k (30 mins gameplay)
                EnvFlavour("Deterministic", lambda rom: {"frameskip": frameskip[rom]}),
                # NoFrameSkip imposes a max episode steps of frameskip * 100k, weird...
                EnvFlavour("NoFrameskip", {"frameskip": 1}),
            ],
        ),
        EnvConfig(
            version="v4",
            kwargs={
                "repeat_action_probability": 0.0,
                "full_action_space": False,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[
                # Unlike v0, v4 has 100k max episode steps
                EnvFlavour("", {"frameskip": (2, 5)}),
                EnvFlavour("Deterministic", lambda rom: {"frameskip": frameskip[rom]}),
                # Same weird frameskip * 100k max steps for v4?
                EnvFlavour("NoFrameskip", {"frameskip": 1}),
            ],
        ),
    ]

    for rom in legacy_games:
        for obs_type in obs_types:
            for config in versions:
                for flavour in config.flavours:
                    name = _rom_id_to_name(rom)
                    name = f"{name}-ram" if obs_type == "ram" else name

                    # Parse config kwargs
                    if callable(config.kwargs):
                        config_kwargs = config.kwargs(rom)
                    else:
                        config_kwargs = config.kwargs

                    # Parse flavour kwargs
                    if callable(flavour.kwargs):
                        flavour_kwargs = flavour.kwargs(rom)
                    else:
                        flavour_kwargs = flavour.kwargs

                    # Register the environment
                    gymnasium.register(
                        id=f"{name}{flavour.suffix}-{config.version}",
                        entry_point="ale_py.env:AtariEnv",
                        kwargs=dict(
                            game=rom,
                            obs_type=obs_type,
                            **config_kwargs,
                            **flavour_kwargs,
                        ),
                    )


def register_v5_envs():
    """Register all v5 environments."""
    all_games = roms.get_all_rom_ids()

    for rom in all_games:
        if rom in {"combat", "joust", "maze_craze", "warlords"}:
            continue

        name = _rom_id_to_name(rom)

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
