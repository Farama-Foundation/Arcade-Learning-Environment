from collections import defaultdict
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from gymnasium.envs.registration import register

ALL_ATARI_GAMES = (
    "adventure",
    "air_raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "atlantis2",
    "backgammon",
    "bank_heist",
    "basic_math",
    "battle_zone",
    "beam_rider",
    "berzerk",
    "blackjack",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "casino",
    "centipede",
    "chopper_command",
    "crazy_climber",
    "crossbow",
    "darkchambers",
    "defender",
    "demon_attack",
    "donkey_kong",
    "double_dunk",
    "earthworld",
    "elevator_action",
    "enduro",
    "entombed",
    "et",
    "fishing_derby",
    "flag_capture",
    "freeway",
    "frogger",
    "frostbite",
    "galaxian",
    "gopher",
    "gravitar",
    "hangman",
    "haunted_house",
    "hero",
    "human_cannonball",
    "ice_hockey",
    "jamesbond",
    "journey_escape",
    "kaboom",
    "kangaroo",
    "keystone_kapers",
    "king_kong",
    "klax",
    "koolaid",
    "krull",
    "kung_fu_master",
    "laser_gates",
    "lost_luggage",
    "mario_bros",
    "miniature_golf",
    "montezuma_revenge",
    "mr_do",
    "ms_pacman",
    "name_this_game",
    "othello",
    "pacman",
    "phoenix",
    "pitfall",
    "pitfall2",
    "pong",
    "pooyan",
    "private_eye",
    "qbert",
    "riverraid",
    "road_runner",
    "robotank",
    "seaquest",
    "sir_lancelot",
    "skiing",
    "solaris",
    "space_invaders",
    "space_war",
    "star_gunner",
    "superman",
    "surround",
    "tennis",
    "tetris",
    "tic_tac_toe_3d",
    "time_pilot",
    "trondead",
    "turmoil",
    "tutankham",
    "up_n_down",
    "venture",
    "video_checkers",
    "video_chess",
    "video_cube",
    "video_pinball",
    "wizard_of_wor",
    "word_zapper",
    "yars_revenge",
    "zaxxon",
)
LEGACY_ATARI_GAMES = (
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
)


class GymFlavour(NamedTuple):
    """A Gymnasium Flavour."""

    suffix: str
    kwargs: Mapping[str, Any] | Callable[[str], Mapping[str, Any]]


class GymConfig(NamedTuple):
    """A Gymnasium Configuration."""

    version: str
    kwargs: Mapping[str, Any]
    flavours: Sequence[GymFlavour]


def _register_configs(
    roms: Sequence[str],
    obs_types: Sequence[str],
    configs: Sequence[GymConfig],
    prefix: str = "",
):
    """Registers all possible configurations of the atari games given a list of roms."""
    for rom in roms:
        for obs_type in obs_types:
            for config in configs:
                for flavour in config.flavours:
                    # convert snake to pascal, ie: space_invaders -> SpaceInvaders
                    name = rom.title().replace("_", "")
                    if obs_type == "ram":
                        name = f"{name}-ram"

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
                    register(
                        id=f"{prefix}{name}{flavour.suffix}-{config.version}",
                        entry_point="ale_py.atari_env:AtariEnv",
                        kwargs={
                            "game": rom,
                            "obs_type": obs_type,
                            **config_kwargs,
                            **flavour_kwargs,
                        },
                    )


def register_gymnasium():
    frameskip: dict[str, int] = defaultdict(lambda: 4, [("space_invaders", 3)])

    configs = [
        GymConfig(
            version="v0",
            kwargs={
                "repeat_action_probability": 0.25,
                "full_action_space": False,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[
                # Default for v0 has 10k steps, no idea why...
                GymFlavour("", {"frameskip": (2, 5)}),
                # Deterministic has 100k steps, close to the standard of 108k (30 mins gameplay)
                GymFlavour("Deterministic", lambda rom: {"frameskip": frameskip[rom]}),
                # NoFrameSkip imposes a max episode steps of frameskip * 100k, weird...
                GymFlavour("NoFrameskip", {"frameskip": 1}),
            ],
        ),
        GymConfig(
            version="v4",
            kwargs={
                "repeat_action_probability": 0.0,
                "full_action_space": False,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[
                # Unlike v0, v4 has 100k max episode steps
                GymFlavour("", {"frameskip": (2, 5)}),
                GymFlavour("Deterministic", lambda rom: {"frameskip": frameskip[rom]}),
                # Same weird frameskip * 100k max steps for v4?
                GymFlavour("NoFrameskip", {"frameskip": 1}),
            ],
        ),
    ]
    _register_configs(LEGACY_ATARI_GAMES, obs_types=("rgb", "ram"), configs=configs)

    # max_episode_steps is 108k frames which is 30 mins of gameplay.
    # This corresponds to 108k / 4 = 27,000 steps
    configs = [
        GymConfig(
            version="v5",
            kwargs={
                "repeat_action_probability": 0.25,
                "full_action_space": False,
                "frameskip": 4,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[GymFlavour("", {})],
        )
    ]
    _register_configs(
        ALL_ATARI_GAMES, obs_types=("rgb", "ram"), configs=configs, prefix="ALE/"
    )
