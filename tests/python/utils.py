"""Utility functions for testing."""

import os
from unittest.mock import patch

import ale_py
import gymnasium
import pytest


@pytest.fixture
def tetris_rom_path():
    """Gets the ROM path of `tetris`."""
    yield os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "resources", "tetris.bin"
    )


@pytest.fixture
def random_rom_path():
    """Gets the ROM path of `random`."""
    yield os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "resources", "random.bin"
    )


@pytest.fixture
def ale():
    """Gets an ALE interface."""
    yield ale_py.ALEInterface()


@pytest.fixture
def tetris(ale, tetris_rom_path):
    """Loads tetris."""
    ale.loadROM(tetris_rom_path)
    yield ale


@pytest.fixture
def tetris_env(request, tetris_rom_path):
    """Pytest fixture for creating a tetris environment."""
    with patch(
        "ale_py.roms.tetris_test", create=True, new_callable=lambda: tetris_rom_path
    ):
        gymnasium.register(
            id="TetrisTest-v0",
            entry_point="ale_py.env:AtariEnv",
            kwargs={"game": "tetris"},
        )

        kwargs = {}
        if hasattr(request, "param") and isinstance(request.param, dict):
            kwargs.update(request.param)

        if hasattr(request, "param"):
            print(request.param)

        yield gymnasium.make("TetrisTest-v0", **kwargs)

        del gymnasium.registry["TetrisTest-v0"]
