#!/usr/bin/env python
# python_example_with_modes.py
# Author: Ben Goodrich & Marlos C. Machado
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceWithModesExample.cpp
import sys
from random import randrange
from ale_python_interface import ALEInterface

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)
# The default is already 0.25, this is just an example
ale.setFloat("repeat_action_probability", 0.25);

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM(sys.argv[1])

#Get the list of available modes and difficulties
avail_modes = ale.getAvailableModes()
avail_diff  = ale.getAvailableDifficulties()

print 'Number of available modes: ', len(avail_modes)
print 'Number of available difficulties: ', len(avail_diff)

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

# Play one episode in each mode and in each difficulty
for mode in avail_modes:
  for diff in avail_diff:

    ale.setDifficulty(diff)
    ale.setMode(mode)
    ale.reset_game()
    print 'Mode {0} difficulty {1}:'.format(mode, diff)

    total_reward = 0
    while not ale.game_over():
      a = legal_actions[randrange(len(legal_actions))]
      # Apply an action and get the resulting reward
      reward = ale.act(a);
      total_reward += reward

    print 'Episode ended with score: ', total_reward
