#!/usr/bin/env python3
"""Microbenchmark of state clone/restore against `reset_game()`.

For each game this reports the per-call cost of `cloneState()`, `cloneSystemState()`,
`restoreState()` and `reset_game()`, and the serialized state size. Restoring a state
cloned right after a reset is the obvious way to make episode resets cheaper, so the
ratio `reset_game / restoreState` is printed as the speed-up such a change could get.

It also checks whether a single snapshot is a faithful substitute for a fresh reset:
after each of several `reset_game()` calls, separated by random play, it compares RAM,
screen, lives and game-over against restoring the first snapshot. `reset_game()` is not
a pure function of the ROM on every game (the RIOT timer is redrawn, some cartridges
re-randomise their RAM, and console RAM carries over between episodes), so games where
this reports a mismatch need more than one snapshot.

`cloneState()` excludes the system RNG, so this does not check that the sticky-action
stream keeps advancing across restores.

Usage:
    ./scripts/microbench-state-clone.py
    ./scripts/microbench-state-clone.py --games breakout,pong --repeats 500 --out clone.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import ale_py
import numpy as np
from ale_py import roms

DEFAULT_GAMES = ("breakout", "pong", "ms_pacman", "space_invaders", "seaquest", "qbert")


def _make(game: str) -> ale_py.ALEInterface:
    """Return a quiet ALE with `game` loaded."""
    ale = ale_py.ALEInterface()
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    ale.setInt("random_seed", 0)
    ale.loadROM(roms.get_rom_path(game))
    return ale


def _timeit(fn: Any, repeats: int) -> dict[str, float]:
    """Time `fn` `repeats` times, returning per-call statistics in microseconds."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": samples[len(samples) // 2],
        "p99_us": samples[min(len(samples) - 1, int(len(samples) * 0.99))],
        "min_us": samples[0],
    }


def _fingerprint(ale: ale_py.ALEInterface) -> dict[str, Any]:
    """Return an observable summary of the emulator's current state."""
    return {
        "ram": ale.getRAM().copy(),
        "screen": ale.getScreen().copy(),
        "lives": ale.lives(),
        "episode_frame": ale.getEpisodeFrameNumber(),
        "game_over": ale.game_over(),
    }


def _diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return the names of the fingerprint fields that differ."""
    out = []
    for key in a:
        left, right = a[key], b[key]
        same = (
            np.array_equal(left, right)
            if isinstance(left, np.ndarray)
            else left == right
        )
        if not same:
            out.append(key)
    return out


def measure_game(game: str, repeats: int, steps_between: int) -> dict[str, Any]:
    """Measure clone/restore/reset cost and check restore-vs-reset equivalence."""
    ale = _make(game)
    # getMinimalActionSet returns `Action` enum members, which numpy cannot index into
    # an integer array; act() takes the enum directly, so keep them as a plain list.
    actions = list(ale.getMinimalActionSet())

    # Cost. `reset_game` is the thing being replaced, so it is the denominator.
    ale.reset_game()
    snapshot = ale.cloneState()

    clone = _timeit(lambda: ale.cloneState(), repeats)
    clone_sys = _timeit(lambda: ale.cloneSystemState(), repeats)
    restore = _timeit(lambda: ale.restoreState(snapshot), repeats)
    reset = _timeit(ale.reset_game, max(20, repeats // 10))
    size_bytes = len(ale.cloneState().serialize())

    # Equivalence: does restoring the snapshot land where a fresh reset does? Both
    # arms start from the same disturbed state, so any difference is attributable to
    # the restore path rather than to history.
    rng = np.random.default_rng(0)
    mismatches: list[dict[str, Any]] = []
    for episode in range(5):
        ale.reset_game()
        after_reset = _fingerprint(ale)

        for _ in range(steps_between):
            ale.act(actions[rng.integers(len(actions))])

        ale.restoreState(snapshot)
        after_restore = _fingerprint(ale)

        # The episode frame counter is expected to differ: `load()` takes it from the
        # snapshot while a reset zeroes it. Report it, but separately from state that
        # would actually change behaviour.
        fields = _diff(after_reset, after_restore)
        behavioural = [f for f in fields if f != "episode_frame"]
        if fields:
            mismatches.append(
                {
                    "episode": episode,
                    "fields": fields,
                    "behavioural": behavioural,
                    "episode_frame": (
                        after_reset["episode_frame"],
                        after_restore["episode_frame"],
                    ),
                }
            )

    equivalent = all(not m["behavioural"] for m in mismatches)
    return {
        "game": game,
        "clone_us": clone,
        "clone_system_us": clone_sys,
        "restore_us": restore,
        "reset_game_us": reset,
        "state_bytes": size_bytes,
        "restore_over_reset": restore["mean_us"] / reset["mean_us"],
        "speedup_if_replaced": reset["mean_us"] / restore["mean_us"],
        "equivalent": equivalent,
        "mismatches": mismatches,
    }


def main() -> None:
    """Run the microbenchmark across the requested games and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=lambda s: s.split(","), default=None)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument(
        "--steps-between",
        type=int,
        default=200,
        help="steps taken between the reset arm and the restore arm",
    )
    parser.add_argument("--out", help="also write the full results to this JSON file")
    args = parser.parse_args()

    games = args.games or list(DEFAULT_GAMES)
    results = [measure_game(g, args.repeats, args.steps_between) for g in games]

    header = (
        f"{'game':<16}{'clone':>10}{'restore':>10}{'reset':>11}"
        f"{'speedup':>10}{'bytes':>9}  same as reset"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['game']:<16}"
            f"{r['clone_us']['mean_us']:>9.1f}u"
            f"{r['restore_us']['mean_us']:>9.1f}u"
            f"{r['reset_game_us']['mean_us'] / 1e3:>10.2f}m"
            f"{r['speedup_if_replaced']:>9.0f}x"
            f"{r['state_bytes']:>9}"
            f"  {'yes' if r['equivalent'] else 'NO'}"
        )

    for r in results:
        for m in r["mismatches"]:
            if m["behavioural"]:
                print(
                    f"  {r['game']}: episode {m['episode']} differs in {m['behavioural']}"
                )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
