/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *  Matthew Hausknecht, Marlos C. Machado and the Reinforcement Learning
 *  and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  sharedLibraryInterfaceExampleWithModes.cpp 
 *
 *  Sample code for running an agent with the shared library interface if one
 *   wants to change the game mode or game difficult (available from v.0.6.0)
 **************************************************************************** */

#include <iostream>
#include <ale_interface.hpp>

#ifdef __USE_SDL
  #include <SDL.h>
#endif

using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " rom_file" << std::endl;
        return 1;
    }

    ALEInterface ale;

    // Get & Set the desired settings
    ale.setInt("random_seed", 123);
    //The default is already 0.25, this is just an example
    ale.setFloat("repeat_action_probability", 0.25);

#ifdef __USE_SDL
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
#endif

    // Load the ROM file. (Also resets the system for new settings to
    // take effect.)
    ale.loadROM(argv[1]);

    // Get the vectors of available modes and difficulties
    ModeVect modes = ale.getAvailableModes();
    DifficultyVect difficulties = ale.getAvailableDifficulties();

    cout << "Number of available modes: " << modes.size() << endl;
    cout << "Number of available difficulties: " << difficulties.size() << endl;

    // Get the vector of legal actions
    ActionVect legal_actions = ale.getLegalActionSet();

    // Play one episode in each mode and in each difficulty
    for (int i = 0; i < modes.size(); i++){
        for (int j = 0; j < difficulties.size(); j++){
            
            ale.setDifficulty(difficulties[j]);
            ale.setMode(modes[i]);
            // Reset after choosing difficulty and mode.
            ale.reset_game();
            cout << "Mode " << modes[i] << ", difficulty " << difficulties[j] << ":" << endl;

            float totalReward = 0;
            while (!ale.game_over()) {
                Action a = legal_actions[rand() % legal_actions.size()];
                // Apply the action and get the resulting reward
                float reward = ale.act(a);
                totalReward += reward;
            }
            cout << "Episode ended with score: " << totalReward << endl;
            ale.reset_game();
        }
    }

    return 0;
}
