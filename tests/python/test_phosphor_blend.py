"""Tests for the phosphor colour-averaging path (`color_averaging`).

The averaging tables are a pure function of the colour palette and are shared between
every environment in the process, so the property sharing has to preserve is: a game's
colour-averaged frames are the same whatever else the process has already emulated.
"""

import subprocess
import sys
import textwrap

import pytest

# The palette is chosen by display format, not by game. breakout and ms_pacman are NTSC
# and share a palette; laser_gates auto-detects as PAL, whose palette maps the same
# indices to different RGB values. A registry that confused the two palettes would
# hand one game the other's tables.
GAMES = ("breakout", "ms_pacman", "laser_gates")

# SHA-256 of every colour-averaged RGB frame of `frame_digest(game)` below, computed
# from a build of `main` before the tables were shared. Pins the output to what
# per-environment tables produced, not only to being self-consistent.
GOLDEN = {
    "breakout": "1b608b8b4fc3d9ba78041ce96ee95686913cf61cfbd46d7cb8ccab31ef262706",
    "ms_pacman": "8c789c8e18fcc1aeebb184c362cf2689275c11743bdea6ad8818370f774d8e63",
    "laser_gates": "3d2bba9e69cca7ec1b359c5426af739bba030ba1bc85acb1a18fca4081003dc2",
}

SCRIPT = textwrap.dedent("""
    import hashlib
    import sys
    import threading

    from ale_py import ALEInterface, LoggerMode, roms

    ALEInterface.setLoggerMode(LoggerMode.Error)


    def frame_digest(game, color_averaging=True, steps=120):
        ale = ALEInterface()
        ale.setBool("color_averaging", color_averaging)
        ale.setInt("random_seed", 0)
        ale.setFloat("repeat_action_probability", 0.0)
        ale.loadROM(roms.get_rom_path(game))

        actions = ale.getMinimalActionSet()
        digest = hashlib.sha256()
        for i in range(steps):
            ale.act(actions[i % len(actions)])
            digest.update(ale.getScreenRGB().tobytes())
            if ale.game_over():
                ale.reset_game()
        return digest.hexdigest()


    mode, games = sys.argv[1], sys.argv[2:]
    if mode == "sequential":
        results = {game: frame_digest(game) for game in games}
    elif mode == "uncoloured":
        results = {game: frame_digest(game, color_averaging=False) for game in games}
    elif mode == "threads":
        results = {}

        def run(index, game):
            results[f"{game}#{index}"] = frame_digest(game)

        threads = [threading.Thread(target=run, args=(i, g)) for i, g in enumerate(games)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    for key, value in results.items():
        print(key, value)
    """)


def digests_in_fresh_process(mode: str, *games: str) -> dict[str, str]:
    """Runs `SCRIPT` in a new interpreter, so the shared tables start out empty."""
    stdout = subprocess.run(
        [sys.executable, "-c", SCRIPT, mode, *games],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return dict(line.split() for line in stdout.splitlines())


@pytest.mark.parametrize("games", [GAMES, GAMES[::-1]], ids=["ntsc-first", "pal-first"])
def test_color_averaging_matches_main(games):
    """Each game's frames match main's, whichever palette's tables were built first."""
    assert digests_in_fresh_process("sequential", *games) == {
        game: GOLDEN[game] for game in games
    }


def test_color_averaging_changes_the_screen():
    """Guards the golden digests against pinning un-averaged frames."""
    uncoloured = digests_in_fresh_process("uncoloured", *GAMES)
    for game in GAMES:
        assert uncoloured[game] != GOLDEN[game]


def test_color_averaging_from_threads():
    """Environments created from several threads at once still get the right tables.

    `loadROM` holds the GIL, so the table builds are serialised rather than truly
    concurrent: this covers registry lookups for both palettes from different threads,
    not a race on the first build. The vector environment never enables colour
    averaging, so no Python API reaches `acquireTables` from parallel native threads.
    """
    games = [game for game in GAMES for _ in range(3)]
    results = digests_in_fresh_process("threads", *games)
    assert results == {f"{game}#{i}": GOLDEN[game] for i, game in enumerate(games)}
