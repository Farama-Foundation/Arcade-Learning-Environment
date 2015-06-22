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
 *  fifoExample.cpp 
 *
 *  Sample code for running a FIFO agent. This interface is provided for 
 *  broader language compatibility; we recommend using the shared interface for
 *  C++ agents.
 **************************************************************************** */

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>

// From RL-Glue agent example.
int randInRange(int max){
	double r, x;
	r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
   	x = (r * (max+1));
	return (int)x;
}


// Print the RAM string
void printRAM(char* str) {

    // First we parse the ram (for pedagogical purposes)
    std::vector<int> ram;
   
    for (int offset = 0; offset < 128; offset++) {

        // Crude but effective
        char buffer[16];
        buffer[0] = str[offset * 2];
        buffer[1] = str[offset * 2 + 1];
        buffer[2] = 0;

        int value = strtol(buffer, NULL, 16);

        ram.push_back(value);
    }

    // Now, if so desired, regurgitate the RAM. 
    const bool printRAM = false;
    
    if (printRAM) for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 16; col++)
            fprintf (stdout, "%2x ", ram[col + row*16]);
        fprintf (stdout, "\n");
    }
}


// Read in RAM and RL data.
bool readData(FILE* alePipe) {

    char buffer[65535];
    fgets(buffer, sizeof(buffer), alePipe);
    
    // Find the first colon, corresponding to the end of the RAM data
    char* endRAM = strchr(buffer, ':');
    printRAM(buffer);

    // Now parse the terminal bit 
    bool terminal = (endRAM[1] == '1');
    
    // Also output reward whenever nonzero
    int reward = strtol(&endRAM[3], NULL, 10);
    if (reward != 0)
        std::cout << "Reward: " << reward << std::endl;

    return terminal;
}


void agentMain(FILE* alePipe) {

    // Read in screen width and height
    char buffer[1024];
    fgets(buffer, sizeof(buffer), alePipe);

    std::cout << "ALE says: " << buffer << std::endl;
    
    // Request RAM & RL data from ALE 
    fputs("0,1,0,1\n", alePipe);

    int frameNumber = 0;

    // Now loop until the episode terminates.
    while (true) {

        // Read in data
        bool terminal = readData(alePipe);

        frameNumber++;

        if (terminal) break;

        // Write back a random action.
        fprintf(alePipe, "%d,%d\n", randInRange(17), 18);
    }

    std::cout << "Episode lasted " << frameNumber << " frames" << std::endl; 
}


int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " rom_file" << std::endl;
        std::cerr << "Note: This example must be run from the same directory as the ALE "
            "executable ('ale')." << std::endl;
        return 1;
    }
    
    std::string romFile(argv[1]);

    // We actually fork two processes, ALE itself and an agent
    std::string aleCmd("./ale -game_controller fifo ");
    aleCmd += romFile;
    
    // Spawn the ALE in read/write mode 
    // We could also use named pipes but that is a bit messier
    FILE* alePipe = popen(aleCmd.c_str(), "r+");
    
    // Now run the agent & communicate with the ale
    agentMain(alePipe);

    pclose(alePipe);
}
