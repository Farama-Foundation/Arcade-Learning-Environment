import pytest

import ale_py


@pytest.fixture
def test_rom_path(resources):
    yield resources["tetris.bin"]


@pytest.fixture
def ale():
    yield ale_py.ALEInterface()


@pytest.fixture
def tetris(ale, test_rom_path):
    ale.loadROM(test_rom_path)
    yield ale
