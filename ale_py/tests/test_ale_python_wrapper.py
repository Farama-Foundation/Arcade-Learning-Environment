import os
from ale_py import ALEInterface


def test_load_rom():
    ale = ALEInterface()
    rom = os.path.join(os.path.dirname(__file__), "fixtures/tetris.bin")
    ale.loadROM(rom)

    assert True
