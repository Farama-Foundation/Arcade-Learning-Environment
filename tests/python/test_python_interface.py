import os
import pickle
import tempfile

import ale_py
import numpy as np
import pytest
from utils import ale, random_rom_path, test_rom_path, tetris  # noqa: F401


def test_ale_version():
    assert hasattr(ale_py, "__version__")


def test_ale_construction(ale):
    assert isinstance(ale, ale_py.ALEInterface)


def test_load_rom(tetris):
    assert isinstance(tetris, ale_py.ALEInterface)


def test_string_config(tetris):
    tetris.setString("record_screen_dir", "/tmp")
    value = tetris.getString("record_screen_dir")
    assert isinstance(value, str)
    assert value == "/tmp"


def test_bool_config(tetris):
    tetris.setBool("sound", False)
    value = tetris.getBool("sound")
    assert isinstance(value, bool)
    assert not value


def test_float_config(tetris):
    tetris.setFloat("repeat_action_probability", 1.0)
    value = tetris.getFloat("repeat_action_probability")
    assert isinstance(value, float)
    assert value == 1.0


def test_int_config(tetris):
    tetris.setInt("frame_skip", 10)
    value = tetris.getInt("frame_skip")
    assert isinstance(value, int)
    assert value == 10


def test_act(tetris):
    enum = tetris.getLegalActionSet()
    tetris.act(enum[0])  # NOOP
    tetris.act(0)  # integer instead of enum


def test_game_over(tetris):
    assert not tetris.game_over()
    while not tetris.game_over():
        tetris.act(0)
    assert tetris.game_over(with_truncation=False)
    assert tetris.game_over(with_truncation=True)
    assert tetris.game_over()


def test_game_over_truncation(ale, test_rom_path):
    ale.setInt("max_num_frames_per_episode", 10)
    ale.setInt("frame_skip", 1)
    ale.loadROM(test_rom_path)
    ale.reset_game()
    for _ in range(10):
        ale.act(0)

    assert ale.getEpisodeFrameNumber() == 10
    assert ale.game_over(with_truncation=True)
    assert not ale.game_over(with_truncation=False)
    assert ale.game_truncated()


def test_get_ram(tetris):
    ram = tetris.getRAM()
    preallocate = np.empty((tetris.getRAMSize()), dtype=np.uint8)
    tetris.getRAM(preallocate)
    assert (ram == preallocate).all()

    preallocate = np.empty((tetris.getRAMSize()), dtype=np.int32)
    with pytest.raises(TypeError) as exc_info:
        tetris.getRAM(preallocate)
    assert exc_info.type == TypeError

    preallocate = np.empty(1, dtype=np.uint8)
    with pytest.raises(RuntimeError) as exc_info:
        tetris.getRAM(preallocate)
    assert exc_info.type == RuntimeError


def test_set_ram(tetris):
    tetris.setRAM(10, 10)
    assert 10 == tetris.getRAM()[10]
    with pytest.raises(RuntimeError):
        tetris.setRAM(1000, 10)
    with pytest.raises(TypeError):
        tetris.setRAM(10, 1000)


def test_reset_game(tetris):
    ram = tetris.getRAM()
    for _ in range(20):
        tetris.act(0)
    tetris.reset_game()
    assert (tetris.getRAM() == ram).all()


def test_get_legal_action_set(tetris):
    action_set = tetris.getLegalActionSet()
    assert len(action_set) == 18


def test_get_minimal_action_set(tetris):
    action_set = tetris.getMinimalActionSet()
    assert len(action_set) < 18


def test_get_available_modes(tetris):
    modes = tetris.getAvailableModes()
    assert len(modes) == 1 and modes[0] == 0


def test_set_mode(tetris):
    with pytest.raises(RuntimeError) as exc_info:
        tetris.setMode(8)
    assert exc_info.type == RuntimeError


def test_get_available_difficulties(tetris):
    modes = tetris.getAvailableDifficulties()
    assert len(modes) == 1 and modes[0] == 0


def test_set_difficulty(tetris):
    with pytest.raises(RuntimeError) as exc_info:
        tetris.setDifficulty(8)
    assert exc_info.type == RuntimeError


def test_get_frame_number(tetris):
    tetris.setInt("frame_skip", 1)
    frame_no = 0
    while not tetris.game_over():
        tetris.act(0)
        frame_no += 1

    tetris.reset_game()
    for _ in range(10):
        frame_no += 1
        tetris.act(0)
    assert tetris.getFrameNumber() == frame_no


def test_lives(tetris):
    assert tetris.lives() == 0


def test_get_episode_frame_number(tetris):
    tetris.setInt("frame_skip", 1)
    for _ in range(10):
        tetris.act(0)
    assert tetris.getEpisodeFrameNumber() == 10


def test_get_screen_dims(tetris):
    dims = tetris.getScreenDims()
    assert isinstance(dims, tuple)
    assert dims == (
        210,
        160,
    )


def test_get_screen_rgb(tetris):
    for _ in range(10):
        tetris.act(0)

    dims = tetris.getScreenDims() + (3,)
    preallocate = np.zeros(dims, dtype=np.uint8)
    tetris.getScreenRGB(preallocate)

    screen = tetris.getScreenRGB()
    assert (preallocate == screen).all()


def test_get_screen(tetris):
    for _ in range(10):
        tetris.act(0)

    dims = tetris.getScreenDims()
    preallocate = np.zeros(dims, dtype=np.uint8)
    tetris.getScreen(preallocate)

    screen = tetris.getScreen()
    assert (preallocate == screen).all()


def test_get_screen_grayscale(tetris):
    for _ in range(10):
        tetris.act(0)

    dims = tetris.getScreenDims()
    preallocate = np.zeros(dims, dtype=np.uint8)
    tetris.getScreenGrayscale(preallocate)

    screen = tetris.getScreenGrayscale()
    assert (preallocate == screen).all()


def test_save_screen_png(tetris):
    for _ in range(10):
        tetris.act(0)

    file = os.path.join(tempfile.gettempdir(), "screen.png")
    tetris.saveScreenPNG(file)
    assert os.path.exists(file)
    os.remove(file)


def test_is_rom_supported(ale, test_rom_path, random_rom_path):
    assert ale.isSupportedROM(test_rom_path)
    assert ale.isSupportedROM(random_rom_path) is None
    with pytest.raises(RuntimeError):
        ale.isSupportedROM("notfound")


def test_clone_restore_state(tetris):
    state = tetris.cloneState()

    for _ in range(10):
        tetris.act(0)

    assert tetris.cloneState() != state
    tetris.restoreState(state)
    assert tetris.cloneState() == state


def test_clone_restore_state_determinism(ale, test_rom_path):
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.loadROM(test_rom_path)
    ale.reset_game()

    num_actions = 100
    action_set = ale.getLegalActionSet()
    seq = list(map(lambda x: x % len(action_set), range(num_actions)))

    def _get_states(seq_):
        states = []
        for action in seq_:
            ale.act(action)
            states.append(ale.cloneState())
            if ale.game_over():
                ale.reset()
        return states

    ale.reset_game()
    first_run = _get_states(seq)
    second_run = _get_states(seq)

    all_equal = True
    for first, second in zip(first_run, second_run):
        all_equal &= first == second
    assert not all_equal


def test_clone_restore_state_stochasticity(ale, test_rom_path):
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.25)
    ale.loadROM(test_rom_path)
    ale.reset_game()

    num_actions = 100
    action_set = ale.getLegalActionSet()
    seq = list(map(lambda x: x % len(action_set), range(num_actions)))

    def _get_states(seq_):
        states = []
        for action in seq_:
            ale.act(action)
            states.append(ale.cloneState())
            if ale.game_over():
                ale.reset()
        return states

    ale.reset_game()
    first_run = _get_states(seq)
    second_run = _get_states(seq)

    all_equal = True
    for first, second in zip(first_run, second_run):
        all_equal &= first == second
    assert not all_equal


def test_clone_restore_state_include_rng(ale, test_rom_path):
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.25)
    ale.loadROM(test_rom_path)
    ale.reset_game()

    num_actions = 100
    action_set = ale.getLegalActionSet()
    seq = list(map(lambda x: x % len(action_set), range(num_actions)))

    def _get_states(seq_):
        states = []
        for action in seq_:
            ale.act(action)
            states.append(ale.cloneState())
            if ale.game_over():
                ale.reset()
        return states

    _get_states(seq[: num_actions // 2])
    without_rng = ale.cloneState()
    with_rng = ale.cloneState(include_rng=True)
    second_half = _get_states(seq[num_actions // 2 :])

    ale.restoreState(without_rng)
    second_half_without_rng = _get_states(seq[num_actions // 2 :])

    ale.restoreState(with_rng)
    second_half_with_rng = _get_states(seq[num_actions // 2 :])

    def _all_equal(first, second):
        all_equal = True
        for first_, second_ in zip(first, second):
            all_equal &= first_ == second_
        return all_equal

    assert _all_equal(second_half, second_half_with_rng)
    assert not _all_equal(second_half, second_half_without_rng)


def test_clone_restore_system_state(ale, test_rom_path):
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.25)
    ale.loadROM(test_rom_path)
    ale.reset_game()

    num_actions = 100
    action_set = ale.getLegalActionSet()
    seq = list(map(lambda x: x % len(action_set), range(num_actions)))
    for action in seq:
        ale.act(action)
        if ale.game_over():
            ale.reset()
    system = ale.cloneSystemState()

    states = []
    for action in seq:
        ale.act(action)
        states.append(ale.cloneSystemState())
        if ale.game_over():
            ale.reset()

    ale.restoreSystemState(system)

    new_states = []
    for action in seq:
        ale.act(action)
        new_states.append(ale.cloneSystemState())
        if ale.game_over():
            ale.reset()

    for first, second in zip(states, new_states):
        assert first == second


def test_state_pickle(tetris):
    for _ in range(10):
        tetris.act(0)

    state = tetris.cloneState()
    file = os.path.join(tempfile.gettempdir(), "ale-state.p")
    with open(file, "wb") as fp:
        pickle.dump(state, fp)

    tetris.reset_game()
    assert tetris.cloneState() != state
    with open(file, "rb") as fp:
        pickeled = pickle.load(fp)
    assert pickeled == state
    tetris.restoreState(pickeled)
    assert tetris.cloneState() == state
    os.remove(file)


@pytest.mark.skipif(ale_py.SDL_SUPPORT is False, reason="SDL is disabled")
def test_display_screen(ale, test_rom_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    ale.setBool("display_screen", True)
    ale.setBool("sound", False)
    ale.loadROM(test_rom_path)
    for _ in range(10):
        ale.act(0)
    del os.environ["SDL_VIDEODRIVER"]
    assert True


def test_set_logger(ale):
    ale.setLoggerMode(ale_py.LoggerMode.Info)
    ale.setLoggerMode(ale_py.LoggerMode.Warning)
    ale.setLoggerMode(ale_py.LoggerMode.Error)
