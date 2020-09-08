# Changelog

These changelogs are for the ALE prior to v0.5. For more recent versions check out the [releases page](https://github.com/mgbellemare/Arcade-Learning-Environment/releases).

## Inter-release notes:
  * color_averaging is now off by default so that environment observations correspond to emulator frames unless requested otherwise.

## ALE 0.5dev_b.
**Released October 4th, 2015.**

  * Enforce flags existence (@mcmachado).
  * Fix RNG issues introduced in 0.5.0.
  * Routines for ALEState serialization (@Jragonmiris). 
  * Additional bug fixes.

## ALE 0.5dev
**Released July 7th, 2015.**

  * Refactored Python getScreenRGB to return unpacked RGB values (@spragunr).
  * Sets the default value of the color_averaging flag to be true. It was true by default in previous versions but was changed in 0.5.0. Reverted for backward compatibility.
  * Added RNG serialization capability.
  * Bug fixes from ALE 0.5.0.

## ALE 0.5.0.
**Released June 22nd, 2015.**

  * Added action_repeat_stochasticity.
  * Added sound playback, visualization.
  * Added screen/sound recording ability.
  * CMake now available.
  * Incorporated Benjamin Goodrich's Python interface.
  * Some game fixes.
  * Added examples for shared library, Python, fifo, RL-Glue interfaces.
  * Incorporated Java agent into main repository.
  * Removed internal controller, now superseded by shared library interface.
  * Better ALEInterface.
  * Many other changes.

## ALE 0.5dev.
**Released February 15th, 2015.**

  * Removed the following command-line flags: 'output_file', 'system_reset_steps', 'use_environment_distribution', 'backward_compatible_save', internal agent flags
  * The flag 'use_starting_actions' was removed and internally its value is always 'true'.
  * The flag 'disable_color_averaging' was renamed to 'color_averaging' and FALSE is its default value.

## ALE 0.4.4.
**Released April 28th, 2014.**

  * Fixed a memory issue in ALEScreen.

## ALE 0.4.3 Bug fix (Mayank Daswani).
**Released April 26th, 2014.**

  * Fixed issues with frame numbers not being correctly updated.
  * Fixed a bug where total reward was not properly reported under frame skipping. 

## ALE 0.4.3.
**Released January 7th, 2013.**

  * Fixed a bug with ALEState's m_frame_number.

## ALE 0.4.2.
**Released June 12th, 2013.**

  * Modified StellaEnvironment save/load interface to provide additional flexibility. 
  * Series of bug fixes from Matthew Hausknecht and community.

## ALE 0.4.1 Bug fix
**Released May 24th, 2013.**

  * Fixed RL-Glue syntax from OBSERVATION to OBSERVATIONS. Thanks to Angus MacIsaac for picking this bug up.

## ALE 0.4.1.
**Released May 22nd, 2013.**

  * Added frame skipping support directly in StellaEnvironment.
  * Reverted default number of episodes to 10.

## ALE 0.4.0.
**Released April 22nd, 2013.**

  * RL-Glue support 
  * Shared library interface
  * Simpler direct environment interfacing
  * Improved environment handling
  * Improved environment customization 
  * Better documentation

## ALE 0.3.1.
**Released August 7th, 2012.**

  * Fixed frames per episode cap for FIFO agents, added functionality to
    limit the total number of frames per run for FIFO agents.

## ALE 0.3.
**Released July 22nd, 2012.**

  * Initial ALE release.
