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
    // Controller settings
    settings.setInt("max_num_frames", 0);
    settings.setInt("max_num_frames_per_episode", 0);

    // FIFO controller settings
    settings.setBool("run_length_encoding", true);

    // Environment customization settings
    settings.setBool("restricted_action_set", false);
    settings.setString("random_seed", "time");
    settings.setBool("color_averaging", false);
    settings.setBool("send_rgb", false);
    settings.setInt("frame_skip", 1);
    settings.setFloat("repeat_action_probability", 0.25);

    // Display Settings
    settings.setBool("display_screen", false);

    // Record settings
    settings.setString("record_sound_filename", "");
}
