"""Backward-equivalence tests for the single-player ALE API.

These tests replay recorded trajectories against a reference dataset generated
by a known-good ALE build and assert bit-identical behaviour (rewards, lives,
RAM, screen, mode metadata, sticky-action RNG, paddle strength, and pickled
``ALEState`` compatibility).

Usage:
    # 1. With a known-good (pre-change) ale_py installed, generate the reference:
    python tests/python/test_backward_equivalence.py --generate

    # 2. Rebuild ale_py from the branch under test, then:
    pytest tests/python/test_backward_equivalence.py -v
"""

from __future__ import annotations

import hashlib
import json
import pickle
import zlib
from pathlib import Path

import ale_py
import numpy as np
import pytest
from ale_py import ALEInterface, roms

DATA_DIR = Path(__file__).parent / "data"
REF_FILE = DATA_DIR / "backward_equivalence_reference.npz"

# Games chosen to cover joystick (pong, space_invaders, freeway, tetris) and
# paddle (breakout, kaboom) controllers, plus every game whose RomSettings
# gained two-player support so their single-player behaviour is pinned to the
# reference build. The reference must be regenerated from a known-good build
# whenever this list changes (see module docstring).
GAMES = [
    "pong",
    "breakout",
    "space_invaders",
    "tetris",
    "freeway",
    "kaboom",
    # games that gained two-player support in the MA-ALE merge
    "boxing",
    "backgammon",
    "double_dunk",
    "entombed",
    "fishing_derby",
    "flag_capture",
    "ice_hockey",
    "mario_bros",
    "othello",
    "space_war",
    "surround",
    "tennis",
    "video_checkers",
    "wizard_of_wor",
]
PADDLE_GAMES = ["breakout", "kaboom"]
# tetris depends on emulator state that cloneState() does not capture (restoring
# the same state into a fresh ALEInterface diverges even on the reference
# build), so it is excluded from the state-restore tests.
STATE_GAMES = [g for g in GAMES if g != "tetris"]
STICKY_PROBS = [0.0, 0.25]
NUM_STEPS = 300
MODE_STEPS = 100
STATE_STEPS = 50
STATE_REPLAY_STEPS = 20
SEED = 123


def quiet():
    ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)


def make_interface(game: str, sticky: float, seed: int = SEED) -> ALEInterface:
    quiet()
    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", sticky)
    ale.loadROM(roms.get_rom_path(game))
    return ale


def run_trajectory(ale: ALEInterface, actions, paddle_strengths=None):
    """Step through `actions` recording everything observable per step."""
    n = len(actions)
    out = {
        "rewards": np.zeros(n, dtype=np.int64),
        "lives": np.zeros(n, dtype=np.int32),
        "terminated": np.zeros(n, dtype=bool),
        "episode_frames": np.zeros(n, dtype=np.int32),
        "ram": np.zeros((n, ale.getRAMSize()), dtype=np.uint8),
        "screen_sha": np.empty(n, dtype="U64"),
    }
    for i, action in enumerate(actions):
        if paddle_strengths is None:
            out["rewards"][i] = ale.act(int(action))
        else:
            out["rewards"][i] = ale.act(int(action), float(paddle_strengths[i]))
        out["lives"][i] = ale.lives()
        out["terminated"][i] = ale.game_over()
        out["episode_frames"][i] = ale.getEpisodeFrameNumber()
        out["ram"][i] = ale.getRAM()
        out["screen_sha"][i] = hashlib.sha256(ale.getScreenRGB().tobytes()).hexdigest()
        if ale.game_over():
            ale.reset_game()
    return out


def stable_seed(*parts) -> int:
    return zlib.crc32("/".join(str(p) for p in parts).encode())


def random_actions(action_set, n, seed):
    rng = np.random.default_rng(seed)
    values = [int(getattr(a, "value", a)) for a in action_set]
    return rng.choice(np.asarray(values, dtype=np.int64), size=n)


TRAJECTORY_KEYS = [
    "rewards",
    "lives",
    "terminated",
    "episode_frames",
    "ram",
    "screen_sha",
]


# ---------------------------------------------------------------------------
# Reference generation
# ---------------------------------------------------------------------------


def generate_reference():
    quiet()
    data: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {"ale_version": ale_py.__version__}

    for game in GAMES:
        print(f"== {game}")
        ale = make_interface(game, sticky=0.0)
        game_meta = {
            "legal_actions": [int(a.value) for a in ale.getLegalActionSet()],
            "minimal_actions": [int(a.value) for a in ale.getMinimalActionSet()],
            "modes": [int(m) for m in ale.getAvailableModes()],
            "difficulties": [int(d) for d in ale.getAvailableDifficulties()],
            "default_mode": int(ale.getMode()) if hasattr(ale, "getMode") else None,
            "screen_dims": list(ale.getScreenDims()),
        }
        meta[game] = game_meta

        # 1. Seeded trajectories with and without sticky actions.
        for sticky in STICKY_PROBS:
            ale = make_interface(game, sticky=sticky)
            actions = random_actions(
                game_meta["minimal_actions"], NUM_STEPS, seed=stable_seed(game, sticky)
            )
            traj = run_trajectory(ale, actions)
            prefix = f"{game}/sticky{sticky}"
            data[f"{prefix}/actions"] = actions
            for k in TRAJECTORY_KEYS:
                data[f"{prefix}/{k}"] = traj[k]

        # 2. Paddle-strength (continuous action) trajectories.
        if game in PADDLE_GAMES:
            ale = make_interface(game, sticky=0.0)
            rng = np.random.default_rng(SEED)
            actions = random_actions(game_meta["minimal_actions"], NUM_STEPS, seed=SEED)
            strengths = rng.uniform(0.0, 1.0, size=NUM_STEPS)
            traj = run_trajectory(ale, actions, paddle_strengths=strengths)
            prefix = f"{game}/paddle"
            data[f"{prefix}/actions"] = actions
            data[f"{prefix}/strengths"] = strengths
            for k in TRAJECTORY_KEYS:
                data[f"{prefix}/{k}"] = traj[k]

        # 3. Per-mode trajectories (catches mode renumbering).
        for mode in game_meta["modes"]:
            ale = make_interface(game, sticky=0.0)
            ale.setMode(mode)
            ale.reset_game()
            actions = random_actions(
                game_meta["minimal_actions"],
                MODE_STEPS,
                seed=stable_seed(game, "mode", mode),
            )
            traj = run_trajectory(ale, actions)
            prefix = f"{game}/mode{mode}"
            data[f"{prefix}/actions"] = actions
            for k in TRAJECTORY_KEYS:
                data[f"{prefix}/{k}"] = traj[k]

        # 4. Serialized ALEState for cross-version pickle compatibility.
        if game not in STATE_GAMES:
            continue
        ale = make_interface(game, sticky=0.0)
        actions = random_actions(game_meta["minimal_actions"], STATE_STEPS, seed=SEED)
        run_trajectory(ale, actions)
        state = ale.cloneState(include_rng=True)
        serialized = pickle.dumps(state)
        replay_actions = random_actions(
            game_meta["minimal_actions"], STATE_REPLAY_STEPS, seed=SEED + 1
        )
        ale.restoreState(state)
        traj = run_trajectory(ale, replay_actions)
        prefix = f"{game}/state"
        data[f"{prefix}/serialized"] = np.frombuffer(serialized, dtype=np.uint8)
        data[f"{prefix}/actions"] = replay_actions
        for k in TRAJECTORY_KEYS:
            data[f"{prefix}/{k}"] = traj[k]

    DATA_DIR.mkdir(exist_ok=True)
    data["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(REF_FILE, **data)
    print(f"\nWrote {REF_FILE} ({REF_FILE.stat().st_size / 1e6:.2f} MB)")
    print(f"Generated with ale_py {ale_py.__version__}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference():
    if not REF_FILE.exists():
        pytest.skip(
            f"Reference data missing. Generate with a known-good ale_py build: "
            f"python {Path(__file__).relative_to(Path.cwd())} --generate"
        )
    with np.load(REF_FILE) as npz:
        data = dict(npz)
    data["meta"] = json.loads(str(data.pop("meta_json")))
    return data


def assert_trajectory_equal(traj, ref, prefix, actions):
    for step, action in enumerate(actions):
        for key in TRAJECTORY_KEYS:
            expected = ref[f"{prefix}/{key}"][step]
            actual = traj[key][step]
            assert np.array_equal(actual, expected), (
                f"{prefix}: first mismatch at step {step} (action={action}) in '{key}': "
                f"expected {expected!r}, got {actual!r}"
            )


@pytest.mark.parametrize("game", GAMES)
def test_static_metadata(reference, game):
    """Action sets, modes, difficulties and default mode must be unchanged."""
    ale = make_interface(game, sticky=0.0)
    ref = reference["meta"][game]
    assert [int(a.value) for a in ale.getLegalActionSet()] == ref["legal_actions"]
    assert [int(a.value) for a in ale.getMinimalActionSet()] == ref["minimal_actions"]
    assert [int(m) for m in ale.getAvailableModes()] == ref["modes"], (
        "getAvailableModes() changed - this breaks gym.make(..., mode=m) for "
        "previously valid modes"
    )
    assert [int(d) for d in ale.getAvailableDifficulties()] == ref["difficulties"]
    if ref["default_mode"] is not None and hasattr(ale, "getMode"):
        assert int(ale.getMode()) == ref["default_mode"]
    assert list(ale.getScreenDims()) == ref["screen_dims"]


@pytest.mark.parametrize("sticky", STICKY_PROBS)
@pytest.mark.parametrize("game", GAMES)
def test_trajectory_equivalence(reference, game, sticky):
    """Bit-identical rewards/lives/RAM/screen for seeded trajectories.

    sticky=0.25 also verifies the action-repeat RNG stream is unchanged.
    """
    prefix = f"{game}/sticky{sticky}"
    actions = reference[f"{prefix}/actions"]
    ale = make_interface(game, sticky=sticky)
    traj = run_trajectory(ale, actions)
    assert_trajectory_equal(traj, reference, prefix, actions)


@pytest.mark.parametrize("game", PADDLE_GAMES)
def test_paddle_strength_equivalence(reference, game):
    """Continuous paddle strength path must be unchanged."""
    prefix = f"{game}/paddle"
    actions = reference[f"{prefix}/actions"]
    strengths = reference[f"{prefix}/strengths"]
    ale = make_interface(game, sticky=0.0)
    traj = run_trajectory(ale, actions, paddle_strengths=strengths)
    assert_trajectory_equal(traj, reference, prefix, actions)


def _mode_params():
    if not REF_FILE.exists():
        return []
    with np.load(REF_FILE) as npz:
        meta = json.loads(str(npz["meta_json"]))
    # Games missing from the reference (e.g. newly added to GAMES before the
    # reference is regenerated) simply contribute no mode cases.
    return [(g, m) for g in GAMES if g in meta for m in meta[g]["modes"]]


@pytest.mark.parametrize("game,mode", _mode_params())
def test_mode_equivalence(reference, game, mode):
    """Every previously-available mode must still be settable and identical."""
    prefix = f"{game}/mode{mode}"
    actions = reference[f"{prefix}/actions"]
    ale = make_interface(game, sticky=0.0)
    ale.setMode(mode)  # raises "Invalid game mode requested" if renumbered
    ale.reset_game()
    traj = run_trajectory(ale, actions)
    assert_trajectory_equal(traj, reference, prefix, actions)


# Games whose RomSettings serialize additional fields since gaining
# multiplayer support. ALEState pickles for these games cannot be exchanged
# with older ALE releases; restoring one must raise a clear error instead of
# silently corrupting the state.
SERIALIZATION_CHANGED_GAMES = [
    "double_dunk",
    "entombed",
    "flag_capture",
    "mario_bros",
    "othello",
    "space_invaders",
    "space_war",
    "tennis",
    "video_checkers",
    "wizard_of_wor",
]


@pytest.mark.parametrize("game", STATE_GAMES)
def test_state_pickle_compatibility(reference, game):
    """ALEState pickles from the reference build must restore correctly.

    This is what pickled envs / saved replay states rely on (uses
    ALEState.__getstate__/__setstate__, i.e. ALEState::serialize()).

    Games whose RomSettings gained serialized fields are a documented
    exception: their cross-version pickles are rejected with a RuntimeError.
    """
    prefix = f"{game}/state"
    serialized = reference[f"{prefix}/serialized"].tobytes()
    actions = reference[f"{prefix}/actions"]
    ale = make_interface(game, sticky=0.0)
    state = pickle.loads(serialized)
    if game in SERIALIZATION_CHANGED_GAMES:
        with pytest.raises(RuntimeError, match="incompatible ALE version"):
            ale.restoreState(state)
        return
    ale.restoreState(state)
    traj = run_trajectory(ale, actions)
    assert_trajectory_equal(traj, reference, prefix, actions)


@pytest.mark.parametrize("game", STATE_GAMES)
def test_clone_restore_roundtrip(game):
    """Same-version clone/restore determinism (no reference data needed)."""
    ale = make_interface(game, sticky=0.0)
    actions = random_actions(ale.getMinimalActionSet(), 30, seed=SEED)
    run_trajectory(ale, actions)
    state = ale.cloneState(include_rng=True)
    replay = random_actions(ale.getMinimalActionSet(), 15, seed=SEED + 1)
    first = run_trajectory(ale, replay)
    ale.restoreState(state)
    second = run_trajectory(ale, replay)
    for key in TRAJECTORY_KEYS:
        assert np.array_equal(first[key], second[key]), key


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate", action="store_true", help="generate reference data"
    )
    args = parser.parse_args()
    if args.generate:
        generate_reference()
    else:
        parser.print_help()
