import warnings
from unittest.mock import patch

import gymnasium
import numpy as np
import pytest
from ale_py.env import AtariEnv
from gymnasium.utils.env_checker import check_env
from utils import tetris_env, tetris_rom_path  # noqa: F401

_VALID_WARNINGS = [
    "we recommend using a symmetric and normalized space",
    "This will error out when the continuous actions are discretized to illegal action spaces",
    "is out of date. You should consider upgrading to version",
]


def test_roms_register():
    registered_roms = [
        env_id
        for env_id, spec in gymnasium.registry.items()
        if spec.entry_point == "ale_py.env:AtariEnv"
    ]

    registered_v0_roms = list(filter(lambda env_id: "v0" in env_id, registered_roms))
    assert len(registered_v0_roms) == 124
    registered_no_frameskip_v0_roms = list(
        filter(lambda env_id: "NoFrameskip-v0" in env_id, registered_roms)
    )
    assert len(registered_no_frameskip_v0_roms) == 62
    registered_v4_roms = list(filter(lambda env_id: "v4" in env_id, registered_roms))
    assert len(registered_v4_roms) == 124
    registered_no_frameskip_v4_roms = list(
        filter(lambda env_id: "NoFrameskip-v4" in env_id, registered_roms)
    )
    assert len(registered_no_frameskip_v4_roms) == 62
    registered_v5_roms = list(filter(lambda env_id: "v5" in env_id, registered_roms))
    assert len(registered_v5_roms) == 104
    assert len(registered_roms) == len(registered_v0_roms) + len(
        registered_v4_roms
    ) + len(registered_v5_roms)


@pytest.mark.parametrize(
    "env_id",
    [
        env_id
        for env_id, spec in gymnasium.registry.items()
        if spec.entry_point == "ale_py.env:AtariEnv"
    ],
)
@pytest.mark.parametrize("continuous", [True, False])
def test_check_env(env_id, continuous):
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = gymnasium.make(env_id, continuous=continuous).unwrapped
        check_env(env, skip_render_check=True)

        env.close()

    for warning in caught_warnings:
        if not any((snippet in warning.message.args[0]) for snippet in _VALID_WARNINGS):
            raise ValueError(warning.message.args[0])


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


def test_gym_keys_to_action():
    env = gymnasium.make("ALE/MsPacman-v5").unwrapped
    assert len(env._action_set) == len(env.get_keys_to_action())
    for keys, action in env.get_keys_to_action().items():
        assert isinstance(keys, tuple)
        assert all(isinstance(key, str) for key in keys)
        assert action in env.action_space
    env.close()

    env = gymnasium.make("ALE/MsPacman-v5", continuous=True).unwrapped
    with pytest.raises(
        AttributeError,
        match="`get_keys_to_action` can't be provided for this Atari environment as `continuous=True`.",
    ):
        env.get_keys_to_action()
    env.close()


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


@pytest.mark.parametrize("tetris_env", [{"continuous": True}], indirect=True)
def test_continuous_action_space(tetris_env):
    assert isinstance(tetris_env.action_space, gymnasium.spaces.Box)
    assert len(tetris_env.action_space.shape) == 1
    assert tetris_env.action_space.shape[0] == 3
    np.testing.assert_array_almost_equal(
        tetris_env.action_space.low, np.array([0.0, -np.pi, 0.0])
    )
    np.testing.assert_array_almost_equal(
        tetris_env.action_space.high, np.array([1.0, np.pi, 1.0])
    )


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
def test_frameskip_warnings(tetris_rom_path, frameskip):
    with patch("ale_py.roms.Tetris", create=True, new_callable=lambda: tetris_rom_path):
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


def test_sound_obs():
    env = gymnasium.make("ALE/MsPacman-v5", sound_obs=True)

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped, skip_render_check=True)

    assert caught_warnings == [], [caught.message.args[0] for caught in caught_warnings]
