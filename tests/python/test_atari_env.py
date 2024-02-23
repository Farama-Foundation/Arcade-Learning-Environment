import warnings
from itertools import product
from unittest.mock import patch

import gymnasium
import numpy as np
import pytest
from ale_py.env import AtariEnv
from ale_py.registration import _register_rom_configs, register_v0_v4_envs
from gymnasium.utils.env_checker import check_env
from utils import test_rom_path, tetris_env  # noqa: F401


def test_register_legacy_env_id():
    prefix = "ALETest/"

    _original_register_gym_configs = _register_rom_configs

    def _mocked_register_gym_configs(*args, **kwargs):
        return _original_register_gym_configs(*args, **kwargs, prefix=prefix)

    with patch(
        "ale_py.registration._register_rom_configs",
        new=_mocked_register_gym_configs,
    ):
        # Register internal IDs
        register_v0_v4_envs()

        # Check if we registered the proper environments
        envids = set(map(lambda e: e.id, gymnasium.registry.values()))
        legacy_games = [
            "Adventure",
            "AirRaid",
            "Alien",
            "Amidar",
            "Assault",
            "Asterix",
            "Asteroids",
            "Atlantis",
            "BankHeist",
            "BattleZone",
            "BeamRider",
            "Berzerk",
            "Bowling",
            "Boxing",
            "Breakout",
            "Carnival",
            "Centipede",
            "ChopperCommand",
            "CrazyClimber",
            "Defender",
            "DemonAttack",
            "DoubleDunk",
            "ElevatorAction",
            "Enduro",
            "FishingDerby",
            "Freeway",
            "Frostbite",
            "Gopher",
            "Gravitar",
            "Hero",
            "IceHockey",
            "Jamesbond",
            "JourneyEscape",
            "Kangaroo",
            "Krull",
            "KungFuMaster",
            "MontezumaRevenge",
            "MsPacman",
            "NameThisGame",
            "Phoenix",
            "Pitfall",
            "Pong",
            "Pooyan",
            "PrivateEye",
            "Qbert",
            "Riverraid",
            "RoadRunner",
            "Robotank",
            "Seaquest",
            "Skiing",
            "Solaris",
            "SpaceInvaders",
            "StarGunner",
            "Tennis",
            "TimePilot",
            "Tutankham",
            "UpNDown",
            "Venture",
            "VideoPinball",
            "WizardOfWor",
            "YarsRevenge",
            "Zaxxon",
        ]
        legacy_games = map(lambda game: f"{prefix}{game}", legacy_games)

        obs_types = ["", "-ram"]
        suffixes = ["Deterministic", "NoFrameskip"]
        versions = ["-v0", "-v4"]

        all_ids = set(
            map("".join, product(legacy_games, obs_types, suffixes, versions))
        )
        assert all_ids.issubset(envids)


def test_register_gym_envs(test_rom_path):
    with patch("ale_py.roms.Tetris", create=True, new_callable=lambda: test_rom_path):
        # Register internal IDs
        # register_v5_envs()

        # Check if we registered the proper environments
        envids = set(map(lambda e: e.id, gymnasium.registry.values()))
        games = ["ALE/Tetris"]

        obs_types = ["", "-ram"]
        suffixes = []
        versions = ["-v5"]

        all_ids = set(map("".join, product(games, obs_types, suffixes, versions)))
        assert all_ids.issubset(envids)


def test_gym_make(tetris_env):
    assert isinstance(tetris_env, gymnasium.Env)


@pytest.mark.parametrize("tetris_env", [{"render_mode": "rgb_array"}], indirect=True)
def test_gym_render_kwarg(tetris_env):
    tetris_env.reset()
    _, _, _, _, info = tetris_env.step(0)
    assert "rgb" not in info
    rgb_array = tetris_env.render()
    assert isinstance(rgb_array, np.ndarray)
    assert rgb_array.shape[-1] == 3


@pytest.mark.parametrize(
    "tetris_env", [{"max_num_frames_per_episode": 10, "frameskip": 1}], indirect=True
)
def test_gym_truncate_on_max_episode_steps(tetris_env):
    tetris_env.reset()

    is_truncated = False
    for _ in range(9):
        _, _, _, is_truncated, _ = tetris_env.step(0)
    assert not is_truncated
    _, _, _, is_truncated, _ = tetris_env.step(0)
    assert is_truncated


@pytest.mark.parametrize("tetris_env", [{"mode": 0, "difficulty": 0}], indirect=True)
def test_gym_mode_difficulty_kwarg(tetris_env):
    pass


@pytest.mark.parametrize("tetris_env", [{"obs_type": "ram"}], indirect=True)
def test_gym_ram_obs(tetris_env):
    tetris_env.reset()
    obs, _, _, _, _ = tetris_env.step(0)
    space = tetris_env.observation_space

    assert isinstance(space, gymnasium.spaces.Box)
    assert np.all(space.low == 0)
    assert np.all(space.high == 255)
    assert space.shape == (128,)

    assert isinstance(obs, np.ndarray)
    assert np.all(obs >= 0) and np.all(obs <= 255)
    assert obs.shape == (128,)


@pytest.mark.parametrize("tetris_env", [{"obs_type": "grayscale"}], indirect=True)
def test_gym_img_grayscale_obs(tetris_env):
    tetris_env.reset()
    obs, _, _, _, _ = tetris_env.step(0)
    space = tetris_env.observation_space

    assert isinstance(space, gymnasium.spaces.Box)
    assert np.all(space.low == 0)
    assert np.all(space.high == 255)
    assert len(space.shape) == 2
    assert space.dtype == np.uint8

    assert isinstance(obs, np.ndarray)
    assert np.all(obs >= 0) and np.all(obs <= 255)
    assert len(obs.shape) == 2


@pytest.mark.parametrize("tetris_env", [{"obs_type": "rgb"}], indirect=True)
def test_gym_img_rgb_obs(tetris_env):
    tetris_env.reset()
    obs, _, _, _, _ = tetris_env.step(0)
    space = tetris_env.observation_space

    assert isinstance(space, gymnasium.spaces.Box)
    assert np.all(space.low == 0)
    assert np.all(space.high == 255)
    assert len(space.shape) == 3
    assert space.shape[-1] == 3
    assert space.dtype == np.uint8

    assert isinstance(obs, np.ndarray)
    assert len(obs.shape) == 3
    assert np.all(obs >= 0) and np.all(obs <= 255)
    assert obs.shape[-1] == 3


@pytest.mark.parametrize("tetris_env", [{"full_action_space": True}], indirect=True)
def test_gym_keys_to_action(tetris_env):
    keys_full_action_space = {
        (None,): 0,
        (32,): 1,
        (119,): 2,
        (100,): 3,
        (97,): 4,
        (115,): 5,
        (100, 119): 6,
        (97, 119): 7,
        (100, 115): 8,
        (97, 115): 9,
        (32, 119): 10,
        (32, 100): 11,
        (32, 97): 12,
        (32, 115): 13,
        (32, 100, 119): 14,
        (32, 97, 119): 15,
        (32, 100, 115): 16,
        (32, 97, 115): 17,
    }
    keys_to_actions = tetris_env.unwrapped.get_keys_to_action()

    assert keys_full_action_space == keys_to_actions


@pytest.mark.parametrize("tetris_env", [{"full_action_space": True}], indirect=True)
def test_gym_action_meaning(tetris_env):
    action_meanings = [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE",
    ]

    assert tetris_env.unwrapped.get_action_meanings() == action_meanings


def test_gym_clone_state(tetris_env):
    tetris_env = tetris_env.unwrapped

    tetris_env.reset(seed=0)
    # Smoke test for clone_state
    tetris_env.step(0)
    state = tetris_env.clone_state()
    for _ in range(100):
        tetris_env.step(tetris_env.action_space.sample())

    tetris_env.restore_state(state)
    assert tetris_env.clone_state() == state


@pytest.mark.parametrize("tetris_env", [{"full_action_space": True}], indirect=True)
def test_gym_action_space(tetris_env):
    assert tetris_env.action_space.n == 18


def test_gym_reset_with_infos(tetris_env):
    pack = tetris_env.reset(seed=0)

    assert isinstance(pack, tuple)
    assert len(pack) == 2

    _, info = pack

    assert isinstance(info, dict)
    assert "seeds" in info
    assert "lives" in info
    assert "episode_frame_number" in info
    assert "frame_number" in info


@pytest.mark.parametrize("frameskip", [0, -1, 4.0, (-1, 5), (0, 5), (5, 2), (1, 2, 3)])
def test_frameskip_warnings(test_rom_path, frameskip):
    with patch("ale_py.roms.Tetris", create=True, new_callable=lambda: test_rom_path):
        with pytest.raises(gymnasium.error.Error):
            AtariEnv("Tetris", frameskip=frameskip)


def test_terminal_signal(tetris_env):
    tetris_env.reset()
    while True:
        _, _, terminal, _, _ = tetris_env.step(tetris_env.action_space.sample())
        emulator_terminal = tetris_env.unwrapped.ale.game_over()
        assert emulator_terminal == terminal
        if terminal:
            break


def test_render_exception(tetris_env):
    tetris_env.reset()

    with pytest.raises(TypeError):
        tetris_env.render(mode="human")

    with pytest.raises(TypeError):
        tetris_env.unwrapped.render(mode="human")


def test_gym_compliance(tetris_env):
    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(tetris_env.unwrapped, skip_render_check=True)

    assert len(caught_warnings) == 0, [w.message for w in caught_warnings]
