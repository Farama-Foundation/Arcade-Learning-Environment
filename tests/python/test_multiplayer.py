"""Tests for the multiplayer (MA-ALE) API: modes, act, lives, and guards."""

import ale_py
import numpy as np
import pytest
from ale_py import ALEInterface, roms

ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)

MULTIPLAYER_ROMS = [
    "backgammon",
    "boxing",
    "combat",
    "double_dunk",
    "entombed",
    "fishing_derby",
    "flag_capture",
    "ice_hockey",
    "joust",
    "mario_bros",
    "maze_craze",
    "othello",
    "pong",
    "space_invaders",
    "space_war",
    "surround",
    "tennis",
    "video_checkers",
    "warlords",
    "wizard_of_wor",
]
# Paddle games support continuous actions (paddle strengths) and up to 4 players
PADDLE_ROMS = ["pong", "warlords"]
# ROMs without any single-player modes: they start in a multiplayer mode
MULTIPLAYER_ONLY_ROMS = ["combat", "maze_craze"]
SINGLE_PLAYER_CAPABLE_ROMS = [
    rom for rom in MULTIPLAYER_ROMS if rom not in MULTIPLAYER_ONLY_ROMS
]

PADDLE_STEPS = 200
SMOKE_STEPS = 300


def make_ale(game, mode=None, seed=42, **settings):
    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    for key, value in settings.items():
        if isinstance(value, bool):
            ale.setBool(key, value)
        elif isinstance(value, int):
            ale.setInt(key, value)
        else:
            ale.setFloat(key, value)
    ale.loadROM(roms.get_rom_path(game))
    if mode is not None:
        ale.setMode(mode)
    ale.reset_game()
    return ale


def available_modes(game):
    """Maps the number of players to the game's modes for that player count."""
    ale = ALEInterface()
    ale.loadROM(roms.get_rom_path(game))
    return {n: list(ale.getAvailableModes(n)) for n in range(1, 5)}


MODES = {game: available_modes(game) for game in MULTIPLAYER_ROMS}
# Every (game, mode, num_players) for 2+ players
MULTIPLAYER_MODES = [
    pytest.param(game, mode, n, id=f"{game}-mode{mode}-{n}p")
    for game in MULTIPLAYER_ROMS
    for n in (2, 3, 4)
    for mode in MODES[game][n]
]
# The first mode for each (game, num_players) with 2+ players
FIRST_MULTIPLAYER_MODES = [
    pytest.param(game, MODES[game][n][0], n, id=f"{game}-{n}p")
    for game in MULTIPLAYER_ROMS
    for n in (2, 3, 4)
    if MODES[game][n]
]


def random_actions(ale, rng, num_players):
    action_set = ale.getMinimalActionSet()
    return [action_set[i] for i in rng.integers(0, len(action_set), num_players)]


@pytest.fixture
def pong():
    return make_ale("pong")


def test_multiplayer_rom_list_complete():
    """MULTIPLAYER_ROMS must list exactly the ROMs with multiplayer modes."""
    found = []
    for rom in roms.get_all_rom_ids():
        ale = ALEInterface()
        ale.loadROM(roms.get_rom_path(rom))
        if any(ale.getAvailableModes(n) for n in (2, 3, 4)):
            found.append(rom)
    assert sorted(found) == sorted(MULTIPLAYER_ROMS)


@pytest.mark.parametrize("game", MULTIPLAYER_ROMS)
def test_mode_lists_disjoint(game):
    """1/2/3/4-player mode lists must not share mode numbers."""
    all_modes = [mode for modes in MODES[game].values() for mode in modes]
    assert len(all_modes) == len(set(all_modes))


@pytest.mark.parametrize("game", MULTIPLAYER_ROMS)
def test_get_available_modes_player_counts(game):
    ale = make_ale(game)
    assert list(ale.getAvailableModes(1)) == list(ale.getAvailableModes())
    assert list(ale.getAvailableModes(3)) == []
    assert MODES[game][2] or MODES[game][4]
    if game in PADDLE_ROMS:
        assert MODES[game][4], "paddle multiplayer games support 4 players"
    else:
        assert MODES[game][4] == [], "joystick games support at most 2 players"
    for invalid in (0, 5, -1):
        with pytest.raises(RuntimeError):
            ale.getAvailableModes(invalid)


@pytest.mark.parametrize("game", MULTIPLAYER_ROMS)
def test_default_mode(game):
    """Games start in a single-player mode unless they have none."""
    ale = make_ale(game)
    if game in MULTIPLAYER_ONLY_ROMS:
        assert MODES[game][1] == []
        assert ale.numPlayersActive() == 2
    else:
        assert ale.numPlayersActive() == 1


@pytest.mark.parametrize("game", MULTIPLAYER_ROMS)
def test_num_players_follows_mode(game):
    ale = make_ale(game)
    for num_players, modes in MODES[game].items():
        for mode in modes:
            ale.setMode(mode)
            assert ale.numPlayersActive() == num_players


@pytest.mark.parametrize("game, mode, num_players", MULTIPLAYER_MODES)
def test_multiplayer_mode_smoke(game, mode, num_players):
    """Every multiplayer mode can be set, reset and played with random actions."""
    ale = make_ale(game, mode)
    assert ale.numPlayersActive() == num_players
    rng = np.random.default_rng(0)
    for _ in range(SMOKE_STEPS):
        rewards = ale.act(random_actions(ale, rng, num_players))
        assert len(rewards) == num_players
        assert len(ale.allLives()) == num_players
        if ale.game_over():
            break


@pytest.mark.parametrize("game, mode, num_players", FIRST_MULTIPLAYER_MODES)
def test_act_size_validation(game, mode, num_players):
    ale = make_ale(game, mode)
    with pytest.raises(RuntimeError):
        ale.act([0] * (num_players - 1))
    with pytest.raises(RuntimeError):
        ale.act([0] * (num_players + 1))
    with pytest.raises(RuntimeError):
        ale.act([0] * num_players, [1.0])  # paddle strength size mismatch


@pytest.mark.parametrize("game, mode, num_players", FIRST_MULTIPLAYER_MODES)
def test_act_rejects_player_b_actions(game, mode, num_players):
    """Multiplayer actions must use the PLAYER_A action range."""
    ale = make_ale(game, mode)
    with pytest.raises(RuntimeError):
        ale.act([0] * (num_players - 1) + [18])  # 18 == PLAYER_B_NOOP


@pytest.mark.parametrize("game, mode, num_players", FIRST_MULTIPLAYER_MODES)
def test_single_act_rejected_in_multiplayer_mode(game, mode, num_players):
    """act(action) only drives player 1, so it must error in multiplayer modes."""
    ale = make_ale(game, mode)
    with pytest.raises(RuntimeError, match="multiplayer mode"):
        ale.act(0)
    with pytest.raises(RuntimeError, match="multiplayer mode"):
        ale.act(0, 0.5)


@pytest.mark.parametrize("game", MULTIPLAYER_ROMS)
def test_lives_api(game):
    ale = make_ale(game)
    for num_players, modes in MODES[game].items():
        if not modes:
            continue
        ale.setMode(modes[0])
        ale.reset_game()
        lives = ale.allLives()
        assert len(lives) == num_players
        if num_players == 1:
            assert ale.lives() == lives[0]
        else:
            with pytest.raises(RuntimeError):
                ale.lives()


def test_two_player_rewards_antisymmetric(pong):
    """Pong is zero-sum: player 2's reward is the negative of player 1's."""
    pong.setMode(pong.getAvailableModes(2)[0])
    pong.reset_game()
    total = np.zeros(2)
    for _ in range(3000):
        rewards = pong.act([3, 3])
        assert rewards[0] == -rewards[1]
        total += rewards
        if pong.game_over():
            break
    assert total[0] != 0, "expected at least one point to be scored"


@pytest.mark.parametrize(
    "game, mode, num_players",
    [p for p in FIRST_MULTIPLAYER_MODES if p.values[0] in PADDLE_ROMS],
)
def test_multiplayer_paddle_strength(game, mode, num_players):
    """Paddle strength must scale movement in multiplayer modes."""

    def ram_after(strengths):
        ale = make_ale(game, mode)
        for _ in range(PADDLE_STEPS):
            ale.act([3] * num_players, strengths)  # RIGHT for every player
        return ale.getRAM().copy()

    full = ram_after([1.0] * num_players)
    full_default = ram_after(None)
    zero = ram_after([0.0] * num_players)
    assert np.array_equal(
        full, full_default
    ), "omitting paddle_strengths must equal strength 1.0"
    assert not np.array_equal(
        full, zero
    ), "strength 0.0 should stop the paddles from moving"


@pytest.mark.parametrize("game, mode, num_players", FIRST_MULTIPLAYER_MODES)
def test_multiplayer_state_roundtrip(game, mode, num_players):
    """clone/restoreState must preserve the player count and MP determinism."""
    ale = make_ale(game, mode, repeat_action_probability=0.25)
    rng = np.random.default_rng(0)
    for _ in range(50):
        ale.act(random_actions(ale, rng, num_players))
    state = ale.cloneState(include_rng=True)
    actions = [random_actions(ale, rng, num_players) for _ in range(20)]
    first = [list(ale.act(a)) for a in actions]
    ram_first = ale.getRAM().copy()
    ale.restoreState(state)
    assert ale.numPlayersActive() == num_players
    second = [list(ale.act(a)) for a in actions]
    assert first == second
    assert np.array_equal(ram_first, ale.getRAM())


@pytest.mark.parametrize("game", MULTIPLAYER_ONLY_ROMS)
def test_vector_env_rejects_multiplayer_modes(game):
    """The vector env steps with single-player APIs; MP modes must error."""
    pytest.importorskip("gymnasium")
    from ale_py.vector_env import AtariVectorEnv

    with pytest.raises(ValueError, match="single-player"):
        AtariVectorEnv(game=game, num_envs=1)


@pytest.mark.parametrize("game", ["breakout"] + SINGLE_PLAYER_CAPABLE_ROMS)
@pytest.mark.parametrize("frameskip", [1, 4])
def test_single_element_act_matches_act(game, frameskip):
    """In single-player modes act([a], [s]) must be identical to act(a, s).

    Uses sticky actions so the RNG stream is exercised, and random paddle
    strengths so paddle games exercise continuous actions.
    """

    def run(step):
        ale = make_ale(
            game, seed=123, frame_skip=frameskip, repeat_action_probability=0.25
        )
        action_set = ale.getMinimalActionSet()
        rng = np.random.default_rng(0)
        rewards = []
        for _ in range(1000):
            action = action_set[rng.integers(0, len(action_set))]
            strength = float(rng.uniform(0.0, 1.0))
            rewards.append(step(ale, action, strength))
            if ale.game_over():
                ale.reset_game()
        return rewards, ale.getRAM().copy(), ale.getScreen().copy()

    single = run(lambda ale, a, s: ale.act(a, s))
    vector = run(lambda ale, a, s: ale.act([a], [s])[0])
    assert single[0] == vector[0]
    assert np.array_equal(single[1], vector[1])
    assert np.array_equal(single[2], vector[2])


@pytest.mark.parametrize("game", SINGLE_PLAYER_CAPABLE_ROMS)
def test_single_element_act_rejects_player_b_actions(game):
    """The PLAYER_B range is rejected for act(list) in every mode."""
    ale = make_ale(game)
    with pytest.raises(RuntimeError):
        ale.act([18])


@pytest.mark.parametrize("game, mode, num_players", FIRST_MULTIPLAYER_MODES)
def test_multiplayer_sound_obs(game, mode, num_players):
    """Multiplayer act must update the audio buffer like single-player act."""
    ale = make_ale(game, mode, sound_obs=True)
    rng = np.random.default_rng(0)
    heard = False
    for _ in range(SMOKE_STEPS):
        ale.act(random_actions(ale, rng, num_players))
        if np.any(ale.getAudio()):
            heard = True
            break
    assert heard, "audio buffer never updated in a multiplayer mode"
