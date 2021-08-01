#!/usr/bin/env python
# python_example_with_modes.py
# Author: Ben Goodrich & Marlos C. Machado
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceWithModesExample.cpp
import sys
from random import randrange
from ale_py import ALEInterface, SDL_SUPPORT

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} rom_file")
    sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt("random_seed", 123)
# The default is already 0.25, this is just an example
ale.setFloat("repeat_action_probability", 0.25)

# Check if we can display the screen
if SDL_SUPPORT:
    ale.setBool("sound", True)
    ale.setBool("display_screen", True)

# Load the ROM file
ale.loadROM(sys.argv[1])

# Get the list of available modes and difficulties
avail_modes = ale.getAvailableModes()
avail_diff = ale.getAvailableDifficulties()

print(f"Number of available modes: {len(avail_modes)}")
print(f"Number of available difficulties: {len(avail_diff)}")

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

# Play one episode in each mode and in each difficulty
for mode in avail_modes:
    for diff in avail_diff:

        ale.setDifficulty(diff)
        ale.setMode(mode)
        ale.reset_game()
        print(f"Mode {mode} difficulty {diff}:")

        total_reward = 0
        while not ale.game_over():
            a = legal_actions[randrange(len(legal_actions))]
            # Apply an action and get the resulting reward
            reward = ale.act(a)
            total_reward += reward

        print(f"Episode ended with score: {total_reward}")
