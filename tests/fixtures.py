import pytest

# Try to import native library before attempting ale_py
try:
    import _ale_py as ale_py
except ImportError:
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
