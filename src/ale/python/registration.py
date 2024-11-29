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


def _register_rom_configs(
    roms: Sequence[str],
    obs_types: Sequence[str],
    configs: Sequence[EnvConfig],
    prefix: str = "",
):
    if len(prefix) > 0 and prefix[-1] != "/":
        prefix += "/"

    for rom in roms:
        for obs_type in obs_types:
            for config in configs:
                for flavour in config.flavours:
                    name = _rom_id_to_name(rom)
                    name = f"{name}-ram" if obs_type == "ram" else name

                    # Parse config kwargs
                    config_kwargs = (
                        config.kwargs(rom) if callable(config.kwargs) else config.kwargs
                    )
                    # Parse flavour kwargs
                    flavour_kwargs = (
                        flavour.kwargs(rom)
                        if callable(flavour.kwargs)
                        else flavour.kwargs
                    )

                    # Register the environment
                    gymnasium.register(
                        id=f"{prefix}{name}{flavour.suffix}-{config.version}",
                        entry_point="ale_py.env:AtariEnv",
                        kwargs=dict(
                            game=rom,
                            obs_type=obs_type,
                            **config_kwargs,
                            **flavour_kwargs,
                        ),
                    )


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

    _register_rom_configs(legacy_games, obs_types, versions)


def register_v5_envs():
    """Register all v5 environments."""
    all_games = roms.get_all_rom_ids()
    obs_types = ["rgb", "ram"]

    # max_episode_steps is 108k frames which is 30 mins of gameplay.
    # This corresponds to 108k / 4 = 27,000 steps
    versions = [
        EnvConfig(
            version="v5",
            kwargs={
                "repeat_action_probability": 0.25,
                "full_action_space": False,
                "frameskip": 4,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[EnvFlavour("", {})],
        )
    ]

    _register_rom_configs(all_games, obs_types, versions, prefix="ALE/")
