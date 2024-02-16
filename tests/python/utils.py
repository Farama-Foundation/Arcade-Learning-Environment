import os
from unittest.mock import patch

import ale_py
import gymnasium
import pytest


@pytest.fixture
def test_rom_path():
    yield os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "resources", "tetris.bin"
    )


@pytest.fixture
def random_rom_path():
    yield os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "resources", "random.bin"
    )


@pytest.fixture
def ale():
    yield ale_py.ALEInterface()


@pytest.fixture
def tetris(ale, test_rom_path):
    ale.loadROM(test_rom_path)
    yield ale


@pytest.fixture
def tetris_env(request, test_rom_path):
    with patch(
        "ale_py.roms.tetris_test", create=True, new_callable=lambda: test_rom_path
    ):
        gymnasium.register(
            id="TetrisTest-v0",
            entry_point="ale_py.env:AtariEnv",
            kwargs={"game": "tetris_test"},
        )

        kwargs = {}
        if hasattr(request, "param") and isinstance(request.param, dict):
            kwargs.update(request.param)

        if hasattr(request, "param"):
            print(request.param)

        yield gymnasium.make("TetrisTest-v0", **kwargs)

        del gymnasium.registry["TetrisTest-v0"]
