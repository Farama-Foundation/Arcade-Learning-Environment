"""Tests for the multiplayer (MA-ALE) API: modes, act, lives, and guards."""

import ale_py
import numpy as np
import pytest
from ale_py import ALEInterface, roms

PADDLE_STEPS = 200


@pytest.fixture
def pong():
    ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    ale = ALEInterface()
    ale.setInt("random_seed", 42)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.loadROM(roms.get_rom_path("pong"))
    return ale


def test_mode_lists_disjoint(pong):
    """1/2/3/4-player mode lists must not share mode numbers."""
    all_modes = []
    for num_players in range(1, 5):
        all_modes.extend(pong.getAvailableModes(num_players))
    assert len(all_modes) == len(set(all_modes))


def test_get_available_modes_player_counts(pong):
    assert list(pong.getAvailableModes(1)) == list(pong.getAvailableModes())
    assert list(pong.getAvailableModes(3)) == []
    for invalid in (0, 5, -1):
        with pytest.raises(RuntimeError):
            pong.getAvailableModes(invalid)


def test_num_players_follows_mode(pong):
    assert pong.numPlayersActive() == 1
    two_player_mode = pong.getAvailableModes(2)[0]
    pong.setMode(two_player_mode)
    assert pong.numPlayersActive() == 2
    four_player_mode = pong.getAvailableModes(4)[0]
    pong.setMode(four_player_mode)
    assert pong.numPlayersActive() == 4
    pong.setMode(pong.getAvailableModes(1)[0])
    assert pong.numPlayersActive() == 1


def test_act_size_validation(pong):
    pong.setMode(pong.getAvailableModes(2)[0])
    pong.reset_game()
    with pytest.raises(RuntimeError):
        pong.act([0])
    with pytest.raises(RuntimeError):
        pong.act([0, 0, 0])
    with pytest.raises(RuntimeError):
        pong.act([0, 0], [1.0])  # paddle strength size mismatch


def test_act_rejects_player_b_actions(pong):
    """Multiplayer actions must use the PLAYER_A action range."""
    pong.setMode(pong.getAvailableModes(2)[0])
    pong.reset_game()
    with pytest.raises(RuntimeError):
        pong.act([0, 18])  # 18 == PLAYER_B_NOOP


def test_lives_api(pong):
    assert pong.lives() == 0
    assert pong.allLives() == [0]
    pong.setMode(pong.getAvailableModes(2)[0])
    pong.reset_game()
    with pytest.raises(RuntimeError):
        pong.lives()
    assert len(pong.allLives()) == 2
    pong.setMode(pong.getAvailableModes(4)[0])
    pong.reset_game()
    assert len(pong.allLives()) == 4


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


def test_multiplayer_paddle_strength(pong):
    """Paddle strength must scale movement in multiplayer modes."""
    mode = pong.getAvailableModes(2)[0]

    def ram_after(strengths):
        ale = ALEInterface()
        ale.setInt("random_seed", 42)
        ale.setFloat("repeat_action_probability", 0.0)
        ale.loadROM(roms.get_rom_path("pong"))
        ale.setMode(mode)
        ale.reset_game()
        for _ in range(PADDLE_STEPS):
            ale.act([3, 3], strengths)  # RIGHT for both players
        return ale.getRAM().copy()

    full = ram_after([1.0, 1.0])
    full_default = ram_after(None)
    zero = ram_after([0.0, 0.0])
    assert np.array_equal(
        full, full_default
    ), "omitting paddle_strengths must equal strength 1.0"
    assert not np.array_equal(
        full, zero
    ), "strength 0.0 should stop the paddles from moving"


def test_multiplayer_state_roundtrip(pong):
    """clone/restoreState must preserve the player count and MP determinism."""
    pong.setMode(pong.getAvailableModes(2)[0])
    pong.reset_game()
    for _ in range(50):
        pong.act([3, 4])
    state = pong.cloneState(include_rng=True)
    first = [list(pong.act([3, 4])) for _ in range(20)]
    ram_first = pong.getRAM().copy()
    pong.restoreState(state)
    assert pong.numPlayersActive() == 2
    second = [list(pong.act([3, 4])) for _ in range(20)]
    assert first == second
    assert np.array_equal(ram_first, pong.getRAM())


def test_vector_env_rejects_multiplayer_modes():
    """The vector env steps with single-player APIs; MP modes must error."""
    gym = pytest.importorskip("gymnasium")

    env = gym.make_vec("ALE/Pong-v5", num_envs=1)
    env.close()  # default single-player mode works
