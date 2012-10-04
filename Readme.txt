This is the 0.3 release of the Arcade Learning Environment (ALE), a platform 
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

-game_controller [fifo|fifo_named|internal] -- specifies how agents interact
  with ALE; see Java agent documentation for details

-config [file] -- specifies a configuration file, from which additional 
  parameters are read

-output_file [file] -- if set, standard output is redirected to the given file.
  Do not use in conjunction with -game_controller fifo_named

-run_length_encoding [false|true] -- determine whether run-length encoding is
  used to send data over pipes; irrelevant when -game_controller internal is 
  set

-max_num_frames_per_episode [n] -- sets the maximum number of frames per
  episode. Once this number is reached, a new episode will start. Currently
  implemented on a per-agent basis with internal agents, or for all
  agents when using pipes (fifo/fifo_named) 

=====================================
Sample agents command-line parameters
=====================================

These parameters are only relevant when using one of the sample agents under
src/agents.

-max_num_episodes [n] -- sets the maximum number of episodes 

-max_num_frames [n] -- sets the maximum number of frames (independent of how 
  many episodes are played)


=====================================
Citing The Arcade Learning Environment: An Evaluation Platform for 
General Agents
=====================================

If you use ALE in your research, we ask that you please cite the following.

M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The arcade learning environment: An evaluation platform for general agents. ArXiv e-prints, July 2012. ArXiv:1207.4708.

In BibTeX format:

@ARTICLE{bellemare12arcade,
  author = {{Bellemare}, M.~G. and {Naddaf}, Y. and {Veness}, J. and {Bowling}, M.},
  title = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1207.4708},
  primaryClass = "cs.AI",
  keywords = {Computer Science - Artificial Intelligence},
  year = 2012,
  month = jul,
}



