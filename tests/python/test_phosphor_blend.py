"""Tests for the phosphor colour-averaging path (`color_averaging`)."""

import hashlib

import pytest
from ale_py import ALEInterface, LoggerMode, roms

# SHA-256 of every colour-averaged RGB frame of `frame_digest(game)` below, computed
# from a build of `main` in which the averaging tables were built unconditionally.
# breakout and ms_pacman are NTSC; laser_gates auto-detects as PAL, a different palette.
GOLDEN = {
    "breakout": "1b608b8b4fc3d9ba78041ce96ee95686913cf61cfbd46d7cb8ccab31ef262706",
    "ms_pacman": "8c789c8e18fcc1aeebb184c362cf2689275c11743bdea6ad8818370f774d8e63",
    "laser_gates": "3d2bba9e69cca7ec1b359c5426af739bba030ba1bc85acb1a18fca4081003dc2",
}


def frame_digest(game: str, color_averaging: bool = True, steps: int = 120) -> str:
    ALEInterface.setLoggerMode(LoggerMode.Error)
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


@pytest.mark.parametrize("game", GOLDEN)
def test_color_averaging_matches_main(game):
    """Building the tables only when needed leaves the averaged frames unchanged."""
    assert frame_digest(game) == GOLDEN[game]


@pytest.mark.parametrize("game", GOLDEN)
def test_color_averaging_changes_the_screen(game):
    """Guards the golden digests against pinning un-averaged frames."""
    assert frame_digest(game, color_averaging=False) != GOLDEN[game]
