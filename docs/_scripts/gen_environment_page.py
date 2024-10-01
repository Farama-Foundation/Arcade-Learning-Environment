import itertools

import ale_py
import gymnasium
import tabulate
from ale_py.registration import _rom_id_to_name
from tqdm import tqdm

gymnasium.register_envs(ale_py)

impossible_roms = {"maze_craze", "joust", "warlords", "combat"}
ALL_ATARI_GAMES = {
    env_spec.kwargs["game"]
    for env_spec in gymnasium.registry.values()
    if isinstance(env_spec.entry_point, str)
    and "ale_py" in env_spec.entry_point
    and env_spec.kwargs["game"] not in impossible_roms
}

# Generate the list of all atari games on atari.md
for rom_id in sorted(ALL_ATARI_GAMES):
    print(f"atari/{rom_id}")


def generate_value_ranges(values):
    for a, b in itertools.groupby(enumerate(values), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]


def shortened_repr(values):
    output = []
    for low, high in generate_value_ranges(values):
        if high - low < 5:
            output.append(", ".join(map(str, range(low, high + 1))))
        else:
            output.append(f"{low}, ..., {high}")
    return "[" + ", ".join(output) + "]"


# # Generate difficult levels table on atari.md
headers = [
    "Environment",
    "Possible Modes",
    "Default Mode",
    "Possible Difficulties",
    "Default Difficulty",
]
rows = []

for rom_id in tqdm(ALL_ATARI_GAMES):
    env_name = _rom_id_to_name(rom_id)

    env = gymnasium.make(f"ALE/{env_name}-v5").unwrapped

    available_difficulties = env.ale.getAvailableDifficulties()
    default_difficulty = env.ale.cloneState().getDifficulty()
    available_modes = env.ale.getAvailableModes()
    default_mode = env.ale.cloneState().getCurrentMode()

    if env_name == "VideoCube":
        available_modes = "[0, 1, 2, 100, 101, 102, ..., 5000, 5001, 5002]"
    else:
        available_modes = shortened_repr(available_modes)

    rows.append(
        [
            env_name,
            available_modes,
            default_mode,
            shortened_repr(available_difficulties),
            default_difficulty,
        ]
    )
    env.close()

print(tabulate.tabulate(rows, headers=headers, tablefmt="github"))
