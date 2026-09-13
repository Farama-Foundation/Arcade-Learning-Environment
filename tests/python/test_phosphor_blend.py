"""Tests for the phosphor colour-averaging path (`color_averaging`)."""

import hashlib
import subprocess
import sys

import pytest
from ale_py import ALEInterface, roms

# Three ROMs whose colour palettes differ, so a table shared between them would show up.
GAMES = ("breakout", "pong", "ms_pacman")

STEPS = 120


def frame_digest(game: str, color_averaging: bool) -> str:
    """Hashes every RGB frame of a short deterministic rollout of `game`."""
    ale = ALEInterface()
    ale.setBool("color_averaging", color_averaging)
    ale.setInt("random_seed", 0)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.loadROM(roms.get_rom_path(game))

    actions = ale.getMinimalActionSet()
    digest = hashlib.sha256()
    for i in range(STEPS):
        ale.act(actions[i % len(actions)])
        digest.update(ale.getScreenRGB().tobytes())
        if ale.game_over():
            ale.reset_game()
    return digest.hexdigest()


def digests_from_subprocess(*games: str) -> dict[str, str]:
    """Returns {game: digest} for `games`, all emulated in one fresh process.

    A fresh process is the point: the shared tables live for the lifetime of the process,
    so within one process every game after the first reuses a registry that an earlier
    game populated.
    """
    stdout = subprocess.run(
        [sys.executable, __file__, *games],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return dict(line.split() for line in stdout.splitlines())


@pytest.fixture(scope="module")
def digests_together():
    """Digests for every game, emulated together in one process."""
    yield digests_from_subprocess(*GAMES)


@pytest.mark.parametrize("game", GAMES)
def test_color_averaging_is_unaffected_by_other_environments(game, digests_together):
    """Colour-averaged frames must not depend on what else the process has emulated.

    The averaging tables are a pure function of (palette, blend ratio) and are shared
    between environments, so this is the property that sharing has to preserve: a game's
    frames are the same whether it built the tables itself or reused another game's.
    """
    alone = digests_from_subprocess(game)[game]
    assert digests_together[game] == alone


@pytest.mark.parametrize("game", GAMES)
def test_color_averaging_changes_the_screen(game):
    """Guards the test above against becoming vacuous if averaging stops being applied."""
    assert frame_digest(game, True) != frame_digest(game, False)


if __name__ == "__main__":
    # Entry point for digests_from_subprocess above.
    for name in sys.argv[1:]:
        print(name, frame_digest(name, True))
