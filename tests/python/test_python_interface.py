import os
import pickle
import struct
import tempfile

import ale_py
import numpy as np
import pytest
from utils import ale, random_rom_path, tetris, tetris_rom_path  # noqa: F401


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


def test_game_over_truncation(ale, tetris_rom_path):
    ale.setInt("max_num_frames_per_episode", 10)
    ale.setInt("frame_skip", 1)
    ale.loadROM(tetris_rom_path)
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
    with pytest.raises(
        TypeError,
        match=r"incompatible function arguments\. The following argument types are supported",
    ):
        tetris.getRAM(preallocate)

    preallocate = np.empty(1, dtype=np.uint8)
    with pytest.raises(
        TypeError,
        match=r"incompatible function arguments\. The following argument types are supported",
    ):
        tetris.getRAM(preallocate)


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


def test_get_sound_obs(ale, tetris_rom_path):
    ale.setBool("sound_obs", True)
    ale.loadROM(tetris_rom_path)
    for _ in range(10):
        ale.act(0)

    preallocate = np.zeros(ale.getAudioSize(), dtype=np.uint8)
    ale.getAudio(preallocate)

    sound = ale.getAudio()
    assert (preallocate == sound).all()


def test_is_rom_supported(ale, tetris_rom_path, random_rom_path):
    assert ale.isSupportedROM(tetris_rom_path)
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


def test_clone_restore_state_determinism(ale, tetris_rom_path):
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.loadROM(tetris_rom_path)
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


def test_clone_restore_state_stochasticity(ale, tetris_rom_path):
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.25)
    ale.loadROM(tetris_rom_path)
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


def test_clone_restore_state_include_rng(ale, tetris_rom_path):
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.25)
    ale.loadROM(tetris_rom_path)
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


def test_clone_restore_system_state(ale, tetris_rom_path):
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.25)
    ale.loadROM(tetris_rom_path)
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


def _reset_tetris(ale, rom_path, color_averaging=False):
    """Loads tetris deterministically and returns a short action stream to replay."""
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setBool("color_averaging", color_averaging)
    ale.loadROM(rom_path)
    ale.reset_game()
    actions = ale.getMinimalActionSet()
    return [actions[i % len(actions)] for i in range(12)]


def _play(ale, stream):
    """Acts through `stream`, returning a copy of the screen and RAM after every step."""
    frames = []
    for action in stream:
        ale.act(action)
        frames.append((ale.getScreen().copy(), ale.getRAM().copy()))
    return frames


def _assert_frames_equal(actual, expected):
    for (screen, ram), (expected_screen, expected_ram) in zip(actual, expected):
        np.testing.assert_array_equal(screen, expected_screen)
        np.testing.assert_array_equal(ram, expected_ram)


_ROUND_TRIPS = {
    "state": (lambda ale: ale.cloneState(), lambda ale, s: ale.restoreState(s)),
    "system_state": (
        lambda ale: ale.cloneSystemState(),
        lambda ale, s: ale.restoreSystemState(s),
    ),
    "pickle": (
        lambda ale: pickle.dumps(ale.cloneState()),
        lambda ale, s: ale.restoreState(pickle.loads(s)),
    ),
    "serialize": (
        lambda ale: ale.cloneState().serialize(),
        lambda ale, s: ale.restoreState(ale_py.ALEState(s)),
    ),
}


@pytest.mark.parametrize("color_averaging", [False, True])
@pytest.mark.parametrize("round_trip", list(_ROUND_TRIPS))
def test_restore_reproduces_screen(ale, tetris_rom_path, round_trip, color_averaging):
    """A restored state must reproduce the screen, not just RAM.

    The state cloned immediately after `reset_game()` is the one a snapshot-based reset
    reuses, and it is the one that used to break: TIA state that reaches nothing but the
    frame buffer (the frame cursor, the object mask pointers, the buffers themselves) was
    left out of the serialized state, so the first frames after a restore rendered wrong.
    With color averaging the previous frame buffer is blended into the screen as well.
    """
    clone, restore = _ROUND_TRIPS[round_trip]
    stream = _reset_tetris(ale, tetris_rom_path, color_averaging)
    state = clone(ale)
    reference = _play(ale, stream)

    restore(ale, state)
    _assert_frames_equal(_play(ale, stream), reference)


def test_repeated_restore_of_one_state(ale, tetris_rom_path):
    """Restoring the *same* state many times must reproduce the same frames every time.

    This is the access pattern of a snapshot-based autoreset. The TIA's frame cursor used
    to carry over from whatever the receiving emulator had last drawn, so how wrong a
    restore was depended on the history before it, not only on the state restored.
    """
    stream = _reset_tetris(ale, tetris_rom_path)[:4]
    state = ale.cloneState()
    reference = _play(ale, stream)

    for _ in range(500):
        ale.restoreState(state)
        _assert_frames_equal(_play(ale, stream), reference)


def _corrupt_inner_state(state, transform):
    """Rebuilds `state` with `transform` applied to the emulator part of its serialization.

    `ALEState.serialize()` is six int32 fields, a length-prefixed emulator state, then two
    more int32 fields; `ALEState(other, emulator_state)` takes the emulator part as is.
    """
    serialized = state.serialize()
    (length,) = struct.unpack_from("<i", serialized, 24)
    inner = serialized[28 : 28 + length]
    return ale_py.ALEState(state, transform(inner))


@pytest.mark.parametrize(
    "transform",
    [
        lambda inner: inner[: len(inner) // 2],
        lambda inner: inner[:-4],
        lambda inner: inner[:-1],
    ],
    ids=["half", "last-field", "last-byte"],
)
def test_restore_corrupted_state_raises(tetris, transform):
    """A truncated state must fail loudly instead of leaving a half-restored emulator."""
    for _ in range(10):
        tetris.act(0)
    state = tetris.cloneState()
    corrupted = _corrupt_inner_state(state, transform)

    with pytest.raises(RuntimeError, match="Failed to restore emulator state"):
        tetris.restoreState(corrupted)


def test_deserialize_truncated_state_raises(tetris):
    serialized = tetris.cloneState().serialize()
    with pytest.raises(RuntimeError, match="truncated or corrupt"):
        ale_py.ALEState(serialized[: len(serialized) // 2])


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
def test_display_screen(ale, tetris_rom_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    ale.setBool("display_screen", True)
    ale.setBool("sound", False)
    ale.loadROM(tetris_rom_path)
    for _ in range(10):
        ale.act(0)
    del os.environ["SDL_VIDEODRIVER"]
    assert True


def test_set_logger(ale):
    ale.setLoggerMode(ale_py.LoggerMode.Info)
    ale.setLoggerMode(ale_py.LoggerMode.Warning)
    ale.setLoggerMode(ale_py.LoggerMode.Error)
