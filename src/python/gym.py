from collections import defaultdict, namedtuple

import ale_py.roms as roms
from ale_py.roms import utils as rom_utils

from gym.envs.registration import register

GymFlavour = namedtuple("GymFlavour", ["suffix", "env_kwargs", "kwargs"])
GymConfig = namedtuple("GymConfig", ["version", "env_kwargs", "flavours"])


def _register_gym_configs(roms, obs_types, configs, prefix=""):
    if len(prefix) > 0 and prefix[-1] != "/":
        prefix += "/"

    for rom in roms:
        for obs_type in obs_types:
            for config in configs:
                for flavour in config.flavours:
                    name = rom_utils.rom_id_to_name(rom)
                    name = f"{name}-ram" if obs_type == "ram" else name

                    # Parse environment kwargs
                    env_kwargs_config = config.env_kwargs
                    if callable(env_kwargs_config):
                        env_kwargs_config = env_kwargs_config(rom)

                    env_kwargs_flavour = flavour.env_kwargs
                    if callable(env_kwargs_flavour):
                        env_kwargs_flavour = env_kwargs_flavour(rom)

                    env_kwargs = dict(
                        game=rom,
                        obs_type=obs_type,
                        **env_kwargs_config,
                        **env_kwargs_flavour,
                    )

                    # Gym kwargs
                    kwargs = flavour.kwargs
                    if callable(kwargs):
                        kwargs = kwargs(rom)

                    assert (
                        "frameskip" in env_kwargs
                    ), "Frameskip required for max_episode_steps"

                    # Register the environment
                    register(
                        id=f"{prefix}{name}{flavour.suffix}-{config.version}",
                        entry_point="gym.envs.atari:AtariEnv",
                        kwargs=env_kwargs,
                        **kwargs,
                    )


def register_legacy_gym_envs():
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
        GymConfig(
            "v0",
            {"repeat_action_probability": 0.25, "full_action_space": False},
            [
                # Default for v0 has 10k steps, no idea why...
                GymFlavour("", {"frameskip": (2, 5)}, {"max_episode_steps": 10000}),
                # Deterministic has 100k steps, close to the standard of 108k (30 mins gameplay)
                GymFlavour(
                    "Deterministic",
                    lambda rom: {"frameskip": frameskip[rom]},
                    {"max_episode_steps": 100000},
                ),
                # NoFrameSkip imposes a max episode steps of frameskip * 100k, weird...
                GymFlavour(
                    "NoFrameskip",
                    {"frameskip": 1},
                    lambda rom: {"max_episode_steps": 100000 * frameskip[rom]},
                ),
            ],
        ),
        GymConfig(
            "v4",
            {"repeat_action_probability": 0.0, "full_action_space": False},
            [
                # Unlike v0, v4 has 100k max episode steps
                GymFlavour("", {"frameskip": (2, 5)}, {"max_episode_steps": 100000}),
                GymFlavour(
                    "Deterministic",
                    lambda rom: {"frameskip": frameskip[rom]},
                    {"max_episode_steps": 100000},
                ),
                # Same weird frameskip * 100k max steps for v4?
                GymFlavour(
                    "NoFrameskip",
                    {"frameskip": 1},
                    lambda rom: {"max_episode_steps": 100000 * frameskip[rom]},
                ),
            ],
        ),
    ]

    _register_gym_configs(legacy_games, obs_types, versions)


def register_gym_envs():
    all_games = list(map(rom_utils.rom_name_to_id, dir(roms)))
    obs_types = ["rgb", "ram"]

    # max_episode_steps is 108k frames which is 30 mins of gameplay.
    # This corresponds to 108k / 4 = 27,000 steps
    versions = [
        GymConfig(
            "v5",
            {
                "repeat_action_probability": 0.25,
                "full_action_space": False,
                "frameskip": 4,
            },
            [GymFlavour("", {}, {"max_episode_steps": 108000 // 4})],
        )
    ]

    _register_gym_configs(all_games, obs_types, versions)
