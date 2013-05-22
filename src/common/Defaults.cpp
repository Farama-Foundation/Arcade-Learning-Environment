/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and 
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  Defaults.cpp 
 *
 *  Defines methods for setting default parameters. 
 *
 **************************************************************************** */
#include "Defaults.hpp"

void setDefaultSettings(Settings &settings) {
    // General settings
    settings.setString("random_seed", "time");

    // Controller settings
    settings.setString("game_controller", "internal");
    settings.setString("player_agent", "random_agent");
    settings.setInt("max_num_episodes", 10);
    settings.setInt("max_num_frames", 0);
    settings.setInt("max_num_frames_per_episode", 0);
    settings.setInt("system_reset_steps", 4);

    // FIFO controller settings
    settings.setBool("run_length_encoding", true);

    // Environment customization settings
    settings.setBool("record_trajectory", false);
    settings.setBool("restricted_action_set", false);
    settings.setBool("use_starting_actions", true);
    settings.setBool("use_environment_distribution", false);
    settings.setString("random_seed", "time");
    settings.setBool("disable_color_averaging", false);
    settings.setInt("frame_skip", 1);

    // Display Settings
    settings.setBool("display_screen", false);
}
