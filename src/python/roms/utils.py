def rom_id_to_name(rom: str) -> str:
    """
    Let the ROM ID be the ROM identifier in snakecase.
        For example, `space_invaders`
    The ROM name is the ROM ID in camelcase.
        For example, `SpaceInvaders`

    This function converts the ROM ID to the ROM name.
        i.e., snakecase -> camelcase
    """
    return rom.title().replace("_", "")


def rom_name_to_id(rom: str) -> str:
    """
    Let the ROM ID be the ROM identifier in snakecase.
        For example, `space_invaders`
    The ROM name is the ROM ID in camelcase.
        For example, `SpaceInvaders`

    This function converts the ROM name to the ROM ID.
        i.e., camelcase -> snakecase
    """
    return "".join(
        map(lambda ch: "_" + ch.lower() if ch.isupper() else ch, rom)
    ).lstrip("_")
