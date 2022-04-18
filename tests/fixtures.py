import pytest

from unittest.mock import patch

from gym.envs.registration import registry, register, make

import ale_py


@pytest.fixture
def test_rom_path(resources):
    yield resources["tetris.bin"]


@pytest.fixture
def random_rom_path(resources):
    yield resources["random.bin"]


@pytest.fixture
def ale():
    yield ale_py.ALEInterface()


@pytest.fixture
def tetris(ale, test_rom_path):
    ale.loadROM(test_rom_path)
    yield ale


@pytest.fixture
def tetris_gym(request, test_rom_path):
    with patch(
        "ale_py.roms.TetrisTest", create=True, new_callable=lambda: test_rom_path
    ):
        register(
            id="TetrisTest-v0",
            entry_point="gym.envs.atari:AtariEnv",
            kwargs={"game": "tetris_test"},
        )

        kwargs = {}
        if hasattr(request, "param") and isinstance(request.param, dict):
            kwargs.update(request.param)

        if hasattr(request, "param"):
            print(request.param)

        yield make("TetrisTest-v0", **kwargs)

        del registry.env_specs["TetrisTest-v0"]
