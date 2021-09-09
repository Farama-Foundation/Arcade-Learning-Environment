import pytest

pytest.importorskip("gym")

import numpy as np

from unittest.mock import patch
from itertools import product

from gym import spaces
from gym.envs.registration import registry
from gym.core import Env

from ale_py.gym.utils import (
    register_legacy_gym_envs,
    register_gym_configs,
    register_gym_envs,
)


def test_register_legacy_env_id():
    prefix = "ALETest/"

    _original_register_gym_configs = register_gym_configs

    def _mocked_register_gym_configs(*args, **kwargs):
        return _original_register_gym_configs(*args, **kwargs, prefix=prefix)

    with patch(
        "ale_py.gym.utils.register_gym_configs", new=_mocked_register_gym_configs
    ):
        # Register internal IDs
        register_legacy_gym_envs()

        # Check if we registered the proper environments
        envids = set(map(lambda e: e.id, registry.all()))
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
        register_gym_envs()

        # Check if we registered the proper environments
        envids = set(map(lambda e: e.id, registry.all()))
        games = ["ALE/Tetris"]

        obs_types = ["", "-ram"]
        suffixes = []
        versions = ["-v5"]

        all_ids = set(map("".join, product(games, obs_types, suffixes, versions)))
        assert all_ids.issubset(envids)


def test_gym_make(tetris_gym):
    assert isinstance(tetris_gym, Env)


@pytest.mark.parametrize("tetris_gym", [{"render_mode": "rgb_array"}], indirect=True)
def test_gym_render_kwarg(tetris_gym):
    tetris_gym.reset()
    _, _, _, info = tetris_gym.step(0)
    assert "rgb" in info
    assert isinstance(info["rgb"], np.ndarray)
    assert info["rgb"].shape[-1] == 3


@pytest.mark.parametrize("tetris_gym", [{"mode": 0, "difficulty": 0}], indirect=True)
def test_gym_mode_difficulty_kwarg(tetris_gym):
    pass


@pytest.mark.parametrize("tetris_gym", [{"obs_type": "ram"}], indirect=True)
def test_gym_ram_obs(tetris_gym):
    tetris_gym.reset()
    obs, _, _, _ = tetris_gym.step(0)
    space = tetris_gym.observation_space

    assert isinstance(space, spaces.Box)
    assert np.all(space.low == 0)
    assert np.all(space.high == 255)
    assert space.shape == (128,)

    assert isinstance(obs, np.ndarray)
    assert np.all(obs >= 0) and np.all(obs <= 255)
    assert obs.shape == (128,)


@pytest.mark.parametrize("tetris_gym", [{"obs_type": "grayscale"}], indirect=True)
def test_gym_img_grayscale_obs(tetris_gym):
    tetris_gym.reset()
    obs, _, _, _ = tetris_gym.step(0)
    space = tetris_gym.observation_space

    assert isinstance(space, spaces.Box)
    assert np.all(space.low == 0)
    assert np.all(space.high == 255)
    assert len(space.shape) == 2
    assert space.dtype == np.uint8

    assert isinstance(obs, np.ndarray)
    assert np.all(obs >= 0) and np.all(obs <= 255)
    assert len(obs.shape) == 2


@pytest.mark.parametrize("tetris_gym", [{"obs_type": "rgb"}], indirect=True)
def test_gym_img_rgb_obs(tetris_gym):
    tetris_gym.reset()
    obs, _, _, _ = tetris_gym.step(0)
    space = tetris_gym.observation_space

    assert isinstance(space, spaces.Box)
    assert np.all(space.low == 0)
    assert np.all(space.high == 255)
    assert len(space.shape) == 3
    assert space.shape[-1] == 3
    assert space.dtype == np.uint8

    assert isinstance(obs, np.ndarray)
    assert len(obs.shape) == 3
    assert np.all(obs >= 0) and np.all(obs <= 255)
    assert obs.shape[-1] == 3


@pytest.mark.parametrize("tetris_gym", [{"full_action_space": True}], indirect=True)
def test_gym_keys_to_action(tetris_gym):
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
    keys_to_actions = tetris_gym.get_keys_to_action()

    assert keys_full_action_space == keys_to_actions


@pytest.mark.parametrize("tetris_gym", [{"full_action_space": True}], indirect=True)
def test_gym_action_meaning(tetris_gym):
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

    assert tetris_gym.get_action_meanings() == action_meanings


@pytest.mark.parametrize("tetris_gym", [{"full_action_space": True}], indirect=True)
def test_gym_action_space(tetris_gym):
    assert tetris_gym.action_space.n == 18
