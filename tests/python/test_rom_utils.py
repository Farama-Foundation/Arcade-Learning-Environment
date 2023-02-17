import pytest
from ale_py.roms.utils import rom_id_to_name, rom_name_to_id


@pytest.mark.parametrize(
    "id, expected",
    [
        ("breakout", "Breakout"),
        ("tic_tac_toe_3d", "TicTacToe3D"),
        ("pitfall2", "Pitfall2"),
        ("video_chess", "VideoChess"),
        ("video_cube", "VideoCube"),
    ],
)
def test_rom_id_to_name(id: str, expected: str):
    assert rom_id_to_name(id) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Breakout", "breakout"),
        ("SpaceInvaders", "space_invaders"),
        ("TicTacToe3D", "tic_tac_toe_3d"),
        ("Pitfall2", "pitfall2"),
        ("VideoChess", "video_chess"),
        ("VideoCube", "video_cube"),
    ],
)
def test_rom_name_to_id(name: str, expected: str):
    assert rom_name_to_id(name) == expected
