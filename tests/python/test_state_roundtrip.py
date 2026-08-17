"""Round-trip fidelity of `cloneState()` / `restoreState()` across every shipped ROM.

The per-ROM `RomSettings` subclasses each hand-roll a `saveState` / `loadState` pair, and
nothing in the type system ties the two halves together: a field written with `putBool`
and read back with `getInt`, or read back in the wrong order, compiles silently and only
shows up as a corrupted game state after a restore. These tests sweep the whole ROM set
so a new or edited game implementation cannot introduce that quietly.
"""

from __future__ import annotations

import pathlib

import ale_py
import numpy as np
import pytest
from ale_py import roms

# Shipped in `ale_py/roms/` but with no `RomSettings` subclass behind them: `loadROM`
# reports "Attempt to wrap ROM ... failed" and calls `std::exit`, which would take the
# pytest process down with it. These are the two-player-only titles.
UNSUPPORTED = frozenset({"combat", "joust", "maze_craze", "warlords"})

ALL_ROMS = sorted(
    path.stem
    for path in pathlib.Path(roms.__file__).parent.glob("*.bin")
    if path.stem not in UNSUPPORTED
)


def _load(game: str, seed: int = 0) -> ale_py.ALEInterface:
    """Loads `game` into a quiet, fully deterministic interface.

    Args:
        game: snake_case name of the ROM to load.
        seed: emulator random seed.

    Returns:
        An `ALEInterface` with the ROM loaded and sticky actions disabled, so that the
        action stream is the only input to the emulation.
    """
    ale = ale_py.ALEInterface()
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.loadROM(roms.get_rom_path(game))
    return ale


def test_all_roms_are_covered():
    """Guards against the sweeps silently shrinking to nothing."""
    assert len(ALL_ROMS) > 100
    assert UNSUPPORTED.isdisjoint(ALL_ROMS)


@pytest.mark.parametrize("game", ALL_ROMS)
def test_serialization_is_a_fixed_point(game):
    """Serializing a state, restoring it and serializing again must not change the bytes.

    This needs no knowledge of what any individual game stores. If `saveState` and
    `loadState` disagree about the type, order or number of the fields they carry, the
    values that come back out are not the values that went in, and the second blob
    differs from the first. It also catches lossy encodings in the emulator devices.
    """
    ale = _load(game)
    actions = ale.getMinimalActionSet()
    ale.reset_game()

    for point in range(20):
        if ale.game_over():
            ale.reset_game()

        before = ale.cloneState().serialize()
        ale.restoreState(ale_py.ALEState(before))
        after = ale.cloneState().serialize()

        assert before == after, (
            f"{game}: restoring a state and re-cloning it produced a different "
            f"serialization at snapshot point {point}"
        )

        for step in range(5):
            ale.act(actions[(point + step) % len(actions)])


@pytest.mark.parametrize("game", ALL_ROMS)
def test_restore_reproduces_the_trajectory(game):
    """A restored state must replay a fixed action stream exactly as the live state did.

    Both arms run on the same emulator object and see the same actions, so any
    difference is attributable to the restore rather than to the reset path.
    """
    ale = _load(game)
    actions = ale.getMinimalActionSet()
    rng = np.random.default_rng(0)
    ale.reset_game()

    for _ in range(17):
        ale.act(actions[rng.integers(len(actions))])

    snapshot = ale.cloneState()
    stream = [actions[rng.integers(len(actions))] for _ in range(200)]

    def roll():
        trace = []
        for action in stream:
            reward = ale.act(action)
            trace.append((ale.getRAM(), reward, ale.lives(), ale.game_over()))
            if ale.game_over():
                break
        return trace

    reference = roll()
    ale.restoreState(snapshot)
    replay = roll()

    assert len(reference) == len(replay), (
        f"{game}: the restored state terminated after {len(replay)} steps, the live "
        f"state after {len(reference)}"
    )

    for step, (ref, alt) in enumerate(zip(reference, replay)):
        ref_ram, ref_rest = ref[0], ref[1:]
        alt_ram, alt_rest = alt[0], alt[1:]
        assert np.array_equal(ref_ram, alt_ram), (
            f"{game}: RAM diverged {step + 1} steps after the restore "
            f"({int((ref_ram != alt_ram).sum())} bytes differ)"
        )
        assert ref_rest == alt_rest, (
            f"{game}: (reward, lives, game_over) diverged {step + 1} steps after the "
            f"restore: {ref_rest} vs {alt_rest}"
        )
