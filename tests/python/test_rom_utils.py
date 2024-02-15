import pytest
from ale_py.roms.utils import rom_id_to_name, rom_name_to_id


@pytest.mark.parametrize(
    "rom_id, expected",
    [
        ("breakout", "Breakout"),
        ("tic_tac_toe_3d", "TicTacToe3D"),
        ("pitfall2", "Pitfall2"),
        ("video_chess", "VideoChess"),
        ("video_cube", "VideoCube"),
    ],
)
def test_rom_id_to_name(rom_id, expected: str):
    assert rom_id_to_name(rom_id) == expected


@pytest.mark.parametrize(
    "rom_name, expected",
    [
        ("Breakout", "breakout"),
        ("SpaceInvaders", "space_invaders"),
        ("TicTacToe3D", "tic_tac_toe_3d"),
        ("Pitfall2", "pitfall2"),
        ("VideoChess", "video_chess"),
        ("VideoCube", "video_cube"),
    ],
)
def test_rom_name_to_id(rom_name, expected: str):
    assert rom_name_to_id(rom_name) == expected
