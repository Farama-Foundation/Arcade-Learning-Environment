import numpy as np

ITERATIONS = 5000


def ale(rom, pybind=True):
    if pybind:
        import ale_py
        _ale = ale_py.ALEInterface()
    else:
        import atari_py
        _ale = atari_py.ALEInterface()
    _ale.setInt("random_seed", 0)
    _ale.setFloat("repeat_action_probability", 0.0)
    _ale.loadROM(rom)
    return _ale


def simulate_preallocate(_ale, steps):
    _ale.reset_game()
    action_set = _ale.getLegalActionSet()

    img = np.zeros((210, 160, 3), dtype=np.uint8)

    for step in range(steps):
        action = np.random.randint(len(action_set))
        _ale.act(action)
        _ale.getScreenRGB(img)
        if _ale.game_over():
            _ale.reset_game()


def simulate(_ale, steps):
    _ale.reset_game()
    action_set = _ale.getLegalActionSet()

    for step in range(steps):
        action = np.random.randint(len(action_set))
        _ale.act(action)
        _ale.getScreenRGB()
        if _ale.game_over():
            _ale.reset_game()


def test_ale_py_sim_no_preallocate(test_rom_path, benchmark):
    np.random.seed(0)
    _ale = ale(test_rom_path, pybind=True)
    benchmark(simulate, _ale, ITERATIONS)


def test_ale_py_sim_preallocate(test_rom_path, benchmark):
    np.random.seed(0)
    _ale = ale(test_rom_path, pybind=True)
    benchmark(simulate_preallocate, _ale, ITERATIONS)


def test_atari_py_sim_preallocate(test_rom_path, benchmark):
    np.random.seed(0)
    _ale = ale(test_rom_path, pybind=False)
    benchmark(simulate_preallocate, _ale, ITERATIONS)


def test_atari_py_sim_no_preallocate(test_rom_path, benchmark):
    np.random.seed(0)
    _ale = ale(test_rom_path, pybind=False)
    benchmark(simulate, _ale, ITERATIONS)
