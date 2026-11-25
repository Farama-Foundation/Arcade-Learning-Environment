#!/usr/bin/env python3
"""Count the distinct start states ``reset_game()`` produces for each ROM.

Used to size the vector environment's reset pool (``reset_pool_size``): a pool
only reproduces the start-state distribution if it holds the states that carry
most of its probability.

What varies between resets
--------------------------
* The emulator's system RNG, which redraws the RIOT timer on every reset (and,
  for some cartridge types, the cartridge RAM). It is seeded from
  ``system_random_seed``, a constant, *not* from ``random_seed``, so the k-th
  reset after a ROM load is the same in every emulator, whatever its seed.
* Console RAM that the ROM does not clear on a soft reset: what the previous
  episode left behind (yars_revenge), or a free-running counter (space_war).
* ``random_seed`` reaches the start state only through sticky actions during
  the episode before the reset.

Method
------
For each ROM and each sticky-action probability (``--sticky``, default 0 and
0.25), ``--resets`` start states are collected and fingerprinted.

* **Rollout.** Only ROMs whose start state depends on the previous episode need
  episodes played between resets, and those are the expensive ones. With
  ``--rollout auto`` (the default), a screening pass collects ``--screen``
  resets with and without a random-policy rollout (sticky 0.25, two seeds). A
  ROM needs rollouts if the rollout pass reaches an observable state the plain
  pass does not. Rollouts use a uniform-random policy, frameskip 4, and are truncated at
  ``--max-steps`` agent steps.
* **Seeds.** Each stream gets a ``random_seed`` drawn from ``--seed``. With
  rollouts, the resets are split across ``--streams`` streams with different
  seeds, because the policy and sticky actions differ per seed. Without
  rollouts, a single long stream is used: the seed cannot reach the start state
  there (the screening pass checks this), and a long stream is what covers the
  reset-index-driven variation.
* **Sticky actions** only act inside ``act()``, never during ``reset_game()``,
  so without rollouts the start states cannot depend on them. Those ROMs are
  collected once and reported for every ``--sticky`` value (the screening pass
  checks this too).
* **Fingerprints.** Each start state is restored into a separate emulator with
  sticky actions off, and a fixed ``--probe-steps`` random action script is
  played from it (each action held for 4 frames).

  - ``observable`` (the headline) hashes the start screen, then each step's
    two max-pooled screens and reward. These are differences an agent can see.
  - ``state`` hashes all of that plus the RAM after every step.
  - ``ram_screen`` hashes the RAM and screen straight after reset.

  The emulator is deterministic, so different hashes are certainly different
  states. Equal hashes mean "indistinguishable under this script", so every
  count is a lower bound.

Statistics (on ``observable``):

* ``distinct``, and ``last_new``, the reset at which the last new state appeared.
* ``unseen``: the Good-Turing estimate ``f1 / n`` of the probability that a new
  reset produces a state never seen (``f1`` = states seen exactly once).
* ``chao1``: an estimate of the total number of states, from ``f1`` and ``f2``.
* ``pool@K``: the expected share of reset probability held by a pool of K
  independent draws, ``sum_s p_s (1 - (1 - p_s)^K)``, with ``p_s`` the observed
  frequencies. This ignores the unseen mass, so cap it with ``1 - unseen``.

Usage
-----
    python scripts/count-reset-states.py breakout space_war --resets 2000
    python scripts/count-reset-states.py --all --jobs 20 --cache reset-cache --json results.json

``--cache`` saves each finished job, so an interrupted run resumes where it
stopped. The results for every ROM are in docs/vector-environment.md, under
"Reset Distribution". The full census takes about 2.5 hours on 22 cores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np
from ale_py import ALEInterface, LoggerMode, roms

KEYS = ("observable", "state", "ram_screen")
POOL_SIZES = (16, 64, 256, 1024, 4096)


def make(game: str, sticky: float, seed: int, max_steps: int = 0) -> ALEInterface:
    """An emulator for ``game`` with frameskip 4, truncated at ``max_steps`` agent steps."""
    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", sticky)
    ale.setInt("frame_skip", 4)  # reset_game() is unaffected; act() repeats 4 frames
    ale.setInt("max_num_frames_per_episode", 4 * max_steps)
    ale.loadROM(roms.get_rom_path(game))
    return ale


class Fingerprint:
    """Fingerprints a start state by replaying it on a separate, non-sticky emulator."""

    def __init__(self, game: str, probe_steps: int):
        """Draws the fixed ``probe_steps``-action script every start state is played with."""
        self.ale = make(game, 0.0, 0)
        self.ale.setInt("frame_skip", 1)
        self.ale.loadROM(roms.get_rom_path(game))
        actions = self.ale.getMinimalActionSet()
        rng = np.random.default_rng(0)
        self.script = [actions[i] for i in rng.integers(len(actions), size=probe_steps)]

    def __call__(self, live: ALEInterface) -> dict[str, str]:
        """The ``observable``, ``state`` and ``ram_screen`` hashes of ``live``'s current state."""
        ram, screen = live.getRAM().tobytes(), live.getScreen().tobytes()
        ram_screen = hashlib.blake2b(ram + screen, digest_size=16)
        observable = hashlib.blake2b(screen, digest_size=16)
        state = hashlib.blake2b(ram + screen, digest_size=16)

        self.ale.restoreState(live.cloneState())
        for action in self.script:
            reward, seen = 0, b""
            for frame in range(4):
                reward += self.ale.act(action)
                if frame >= 2:  # the two frames the agent max-pools
                    seen += self.ale.getScreen().tobytes()
            seen += int(reward).to_bytes(8, "little", signed=True)
            observable.update(seen)
            state.update(seen + self.ale.getRAM().tobytes())
            if self.ale.game_over():
                break
        return {
            "observable": observable.hexdigest(),
            "state": state.hexdigest(),
            "ram_screen": ram_screen.hexdigest(),
        }


def collect(
    game: str,
    sticky: float,
    seed: int,
    resets: int,
    rollout: bool,
    max_steps: int,
    probe_steps: int,
) -> list[dict[str, str]]:
    """Fingerprints of ``resets`` successive start states of one emulator."""
    ale = make(game, sticky, seed, max_steps)
    actions = ale.getMinimalActionSet()
    policy = np.random.default_rng(seed)
    fp = Fingerprint(game, probe_steps)
    out = []
    for _ in range(resets):
        ale.reset_game()
        out.append(fp(ale))
        if rollout:
            while not ale.game_over():  # game over or truncated at max_steps
                ale.act(actions[policy.integers(len(actions))])
    return out


def screen(game: str, args, seeds: list[int]) -> dict | None:
    """Decide whether ``game`` needs rollouts, and whether the seed matters without them."""
    try:
        plain = [
            collect(game, 0.25, s, args.screen, False, 0, args.probe_steps)
            for s in seeds
        ]
        plain_no_sticky = collect(
            game, 0.0, seeds[0], args.screen, False, 0, args.probe_steps
        )
        played = [
            collect(game, 0.25, s, args.screen, True, args.max_steps, args.probe_steps)
            for s in seeds
        ]
    except RuntimeError as e:  # ROM file missing or unsupported
        print(f"skipping {game}: {e}", flush=True)
        return None
    plain_set = {d["observable"] for p in plain for d in p}
    new = sum(d["observable"] not in plain_set for p in played for d in p)
    return {
        "game": game,
        "rollout_new_share": new / sum(map(len, played)),
        "seed_matters_without_rollout": plain[0] != plain[1],
        "sticky_matters_without_rollout": plain[0] != plain_no_sticky,
    }


def run_cached(pool, jobs: list[tuple], cache: str | None, label: str) -> list:
    """Run ``(fn, args)`` jobs on ``pool``, saving each result as soon as it finishes.

    A rerun with the same ``cache`` directory picks up the saved results and
    runs only the missing jobs.
    """

    def path(job):
        fn, args = job
        name = "-".join(
            (
                f"{a.screen}-{a.probe_steps}-{a.max_steps}"
                if isinstance(a, argparse.Namespace)
                else "_".join(map(str, a)) if isinstance(a, list) else str(a)
            )
            for a in args
        )
        return os.path.join(cache, f"{fn.__name__}-{name}.json") if cache else None

    results = [None] * len(jobs)
    todo = []
    for i, job in enumerate(jobs):
        if cache and os.path.exists(path(job)):
            with open(path(job)) as f:
                results[i] = json.load(f)
        else:
            todo.append(i)
    start, done = time.time(), 0
    print(
        f"{label}: {len(jobs) - len(todo)} cached, {len(todo)} to run",
        file=sys.stderr,
        flush=True,
    )
    for i, result in pool.imap_unordered(_indexed, [(i, jobs[i]) for i in todo]):
        results[i] = result
        if cache:
            with open(path(jobs[i]) + ".tmp", "w") as f:
                json.dump(result, f)
            os.replace(path(jobs[i]) + ".tmp", path(jobs[i]))
        done += 1
        elapsed = time.time() - start
        print(
            f"{label}: {done}/{len(todo)} after {elapsed / 60:.1f} min "
            f"(started-order ETA {elapsed / done * (len(todo) - done) / 60:.0f} min)",
            file=sys.stderr,
            flush=True,
        )
    return results


def _indexed(item):
    i, (fn, args) = item
    return i, fn(*args)


def summarise(draws: list[dict[str, str]]) -> dict:
    """Distinct counts per fingerprint, and saturation statistics on ``observable``."""
    out = {
        "resets": len(draws),
        "distinct": {k: len({d[k] for d in draws}) for k in KEYS},
    }
    obs = [d["observable"] for d in draws]
    seen, last_new = set(), 0
    for i, o in enumerate(obs):
        if o not in seen:
            seen.add(o)
            last_new = i + 1
    counts = np.array(list(Counter(obs).values()))
    f1, f2, n = int((counts == 1).sum()), int((counts == 2).sum()), len(obs)
    p = counts / n
    out |= {
        "last_new": last_new,
        "unseen": f1 / n,
        "chao1": len(counts) + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2),
        "pool": {k: float(np.sum(p * (1 - (1 - p) ** k))) for k in POOL_SIZES},
    }
    return out


def main() -> None:
    """Screen, collect and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("games", nargs="*", help="snake_case ROM names")
    parser.add_argument("--all", action="store_true", help="every supported ROM")
    parser.add_argument(
        "--resets",
        type=int,
        default=10_000,
        help="start states per ROM and sticky value",
    )
    parser.add_argument("--sticky", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument(
        "--rollout", choices=["auto", "always", "never"], default="auto"
    )
    parser.add_argument(
        "--screen", type=int, default=300, help="resets per screening pass"
    )
    parser.add_argument(
        "--streams", type=int, default=4, help="seeds to split rollout resets over"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="rollout truncation, agent steps"
    )
    parser.add_argument("--probe-steps", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=2026, help="draws every stream's random_seed"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--json", help="write the results to this file")
    parser.add_argument(
        "--cache", help="save each finished job here; a rerun resumes from it"
    )
    args = parser.parse_args()

    games = roms.get_all_rom_ids() if args.all else args.games
    if not games:
        parser.error("name some ROMs or pass --all")
    # combat, joust, maze_craze and warlords have no single-player environment.
    # loadROM calls exit() on them, which would kill
    # a pool worker and hang the pool: drop those up front.
    unsupported = [
        g for g in games if not ALEInterface.isSupportedROM(roms.get_rom_path(g))
    ]
    if unsupported:
        print(
            f"skipping ROMs with no single-player environment: {', '.join(unsupported)}",
            file=sys.stderr,
        )
        games = [g for g in games if g not in unsupported]
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)
    seeds = [
        int(s)
        for s in np.random.SeedSequence(args.seed).generate_state(2 + args.streams)
        & 0x7FFFFFFF
    ]
    screen_seeds, stream_seeds = seeds[:2], seeds[2:]

    with mp.Pool(args.jobs) as pool:
        if args.rollout == "auto":
            screened = run_cached(
                pool,
                [(screen, (g, args, screen_seeds)) for g in games],
                args.cache,
                "screen",
            )
            screened = {s["game"]: s for s in screened if s is not None}
        else:
            screened = {
                g: {
                    "game": g,
                    "rollout_new_share": float(args.rollout == "always"),
                    "seed_matters_without_rollout": None,
                    "sticky_matters_without_rollout": None,
                }
                for g in games
            }
        jobs, keys = [], []
        for game, s in screened.items():
            rollout = s["rollout_new_share"] > 0
            streams = stream_seeds if rollout else stream_seeds[:1]
            # Without rollouts the sticky value cannot matter: collect once.
            for sticky in (
                args.sticky
                if rollout or s["sticky_matters_without_rollout"]
                else ["any"]
            ):
                for seed in streams:
                    jobs.append(
                        (
                            game,
                            0.25 if sticky == "any" else sticky,
                            seed,
                            args.resets // len(streams),
                            rollout,
                            args.max_steps,
                            args.probe_steps,
                        )
                    )
                    keys.append((game, sticky, rollout))
        # Longest jobs (the rollout streams) first, so they do not straggle at the end.
        order = sorted(range(len(jobs)), key=lambda i: not jobs[i][4])
        jobs, keys = [jobs[i] for i in order], [keys[i] for i in order]
        draws = run_cached(pool, [(collect, j) for j in jobs], args.cache, "collect")

    merged: dict[tuple, list] = {}
    for key, d in zip(keys, draws):
        merged.setdefault(key, []).extend(d)
    results = [
        {"game": g, "sticky": st, "rollout": ro, **screened[g], **summarise(d)}
        for (g, st, ro), d in merged.items()
    ]

    print(
        "| ROM | sticky | rollout | resets | observable | state | RAM+screen | last new | unseen | Chao1 | "
        + " | ".join(f"pool@{k}" for k in POOL_SIZES)
        + " |"
    )
    print(
        "|---|---:|:-:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(POOL_SIZES)
    )
    for r in sorted(
        results,
        key=lambda r: (-r["distinct"]["observable"], r["game"], str(r["sticky"])),
    ):
        d = r["distinct"]
        pools = " | ".join(f"{v:.3f}" for v in r["pool"].values())
        print(
            f"| {r['game']} | {r['sticky']} | {'yes' if r['rollout'] else 'no'} | {r['resets']} | "
            f"{d['observable']} | {d['state']} | {d['ram_screen']} | {r['last_new']} | "
            f"{r['unseen']:.4f} | {r['chao1']:.0f} | {pools} |"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"args": vars(args), "seeds": seeds, "results": results}, f, indent=1
            )


if __name__ == "__main__":
    main()
