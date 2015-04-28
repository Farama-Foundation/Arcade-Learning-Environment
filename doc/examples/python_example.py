#!/usr/bin/env python

# ale_python_test1.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from ALE provided in
# doc/examples/sharedLibraryInterfaceExample.cpp

import sys
from ale_python_interface import ALEInterface
import numpy as np

if len(sys.argv) < 2:
  print("Usage ./ale_python_test1.py <ROM_FILE_NAME>")
  sys.exit()

ale = ALEInterface()

max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
ale.setInt("random_seed",123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
  ale.setBool("display_screen",True)
  ale.setBool("sound",True)

random_seed = ale.getInt("random_seed")
print("random_seed: " + str(random_seed))

ale.loadROM(sys.argv[1])
legal_actions = ale.getLegalActionSet()

for episode in range(10):
  total_reward = 0.0
  while not ale.game_over():
    a = legal_actions[np.random.randint(legal_actions.size)]
    reward = ale.act(a);
    total_reward += reward
  print("Episode " + str(episode) + " ended with score: " + str(total_reward))
  ale.reset_game()
