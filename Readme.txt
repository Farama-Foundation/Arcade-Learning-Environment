This is the 0.5 release of the Arcade Learning Environment (ALE), a platform 
designed for AI research. ALE is based on Stella, an Atari 2600 VCS emulator. 
More information and ALE-related publications can be found at

http://www.arcadelearningenvironment.org

We encourage you to use the Arcade Learning Environment in your research. In
return, we would appreciate if you cited ALE in publications that rely on
it (BibTeX entry at the end of this document).

Feedback and suggestions are welcome and may be addressed to any active member 
of the ALE team.

Enjoy,
The ALE team

===============================
List of command-line parameters
===============================

Execute ./ale -help for more details; alternatively, see documentation 
available at http://www.arcadelearningenvironment.org.

-random_seed [n] -- sets the random seed; defaults to the current time

-game_controller [fifo|fifo_named] -- specifies how agents interact
  with ALE; see Java agent documentation for details

-config [file] -- specifies a configuration file, from which additional 
  parameters are read

-run_length_encoding [false|true] -- determine whether run-length encoding is
  used to send data over pipes; irrelevant when an internal agent is 
  being used

-max_num_frames_per_episode [n] -- sets the maximum number of frames per
  episode. Once this number is reached, a new episode will start. Currently
  implemented for all agents when using pipes (fifo/fifo_named) 

-max_num_frames [n] -- sets the maximum number of frames (independent of how 
  many episodes are played)


=====================================
Citing The Arcade Learning Environment: An Evaluation Platform for 
General Agents
=====================================

If you use ALE in your research, we ask that you please cite the following.

M. G. Bellemare, Y. Naddaf, J. Veness and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013.

In BibTeX format:

@ARTICLE{bellemare13arcade,
  author = {{Bellemare}, M.~G. and {Naddaf}, Y. and {Veness}, J. and {Bowling}, M.},
  title = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
  journal = {Journal of Artificial Intelligence Research},
  year = "2013",
  month = "jun",
  volume = "47",
  pages = "253--279",
}



