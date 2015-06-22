/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *  Matthew Hausknecht, and the Reinforcement Learning and Artificial Intelligence 
 *  Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  videoRecordingExample.cpp 
 *
 *  An example on recording video with the ALE. This requires SDL. 
 *  See manual for details. 
 **************************************************************************** */

#include <iostream>
#include <ale_interface.hpp>
#include <cstdlib>

#ifndef __USE_SDL
#error Video recording example is disabled as it requires SDL. Recompile with -DUSE_SDL=ON. 
#else

#include <SDL.h>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " rom_file" << std::endl;
        return 1;
    }

    ALEInterface ale;

    // Get & Set the desired settings
    ale.setInt("random_seed", 123);

    // We enable both screen and sound, which we will need for recording. 
    ale.setBool("display_screen", true);
    // You may leave sound disabled (by setting this flag to false) if so desired. 
    ale.setBool("sound", true);

    std::string recordPath = "record";
    std::cout << std::endl;

    // Set record flags
    ale.setString("record_screen_dir", recordPath.c_str());
    ale.setString("record_sound_filename", (recordPath + "/sound.wav").c_str());
    // We set fragsize to 64 to ensure proper sound sync 
    ale.setInt("fragsize", 64);

    // Not completely portable, but will work in most cases
    std::string cmd = "mkdir ";
    cmd += recordPath; 
    system(cmd.c_str());

    // Load the ROM file. (Also resets the system for new settings to
    // take effect.)
    ale.loadROM(argv[1]);

    // Get the vector of legal actions
    ActionVect legal_actions = ale.getLegalActionSet();

    // Play a single episode, which we record. 
    while (!ale.game_over()) {
        
        Action a = legal_actions[rand() % legal_actions.size()];
        // Apply the action (discard the resulting reward) 
        ale.act(a);
    }

    std::cout << std::endl;
    std::cout << "Recording complete. To create a video, you may want to run \n"
        "  doc/scripts/videoRecordingExampleJoinXXX.sh. See manual for details.." << std::endl;

    return 0;
}
#endif // __USE_SDL
