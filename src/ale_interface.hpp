#ifndef ALE_INTERFACE_H
#define ALE_INTERFACE_H

#include <cstdlib>
#include <ctime>
#include "emucore/m6502/src/bspf/src/bspf.hxx"
#include "emucore/Console.hxx"
#include "emucore/Event.hxx"
#include "emucore/PropsSet.hxx"
#include "emucore/Settings.hxx"
#include "emucore/FSNode.hxx"
#include "emucore/OSystem.hxx"
#include "os_dependent/SettingsUNIX.hxx"
#include "os_dependent/OSystemUNIX.hxx"
#include "control/fifo_controller.h"
#include "control/internal_controller.h"
#include "common/Constants.h"
#include "common/Defaults.hpp"
#include "games/RomSettings.hpp"
#include "games/Roms.hpp"
#include "agents/PlayerAgent.hpp"

static const std::string Version = "0.3";

/* display welcome message */
static std::string welcomeMessage() {

    // ALE welcome message
    std::ostringstream oss;

    oss << "A.L.E: Arcade Learning Environment (version "
        << Version << ")\n" 
        << "[Powered by Stella]\n"
        << "Use -help for help screen.";

    return oss.str();
}


/**
   This class interfaces ALE with external code for controlling agents.
 */
class ALEInterface
{
public:
    OSystem* theOSystem;
    InternalController* game_controller;
    MediaSource *mediasrc;
    System* emulator_system;
    RomSettings* game_settings;

    int screen_width, screen_height;  // Dimensions of the screen
    IntMatrix screen_matrix;     // This contains the raw pixel representation of the screen
    IntVect ram_content;         // This contains the ram content of the Atari

    int frame;                   // Current frame number
    int max_num_frames;          // Maximum number of frames allowed in this episode
    float game_score;            // Score accumulated throughout the course of a game
    ActionVect legal_actions;    // Vector of allowed actions for this game
    ActionVect minimal_actions;  // Vector of minimal actions for this game
    Action last_action;          // Always stores the latest action taken
    time_t time_start, time_end; // Used to keep track of fps
    bool display_active;         // Should the screen be displayed or not

public:
    ALEInterface(): theOSystem(NULL), game_controller(NULL), mediasrc(NULL), emulator_system(NULL),
                    game_settings(NULL), frame(0), max_num_frames(-1),
                    game_score(0), display_active(false) {
    }

    ~ALEInterface() {
        if (theOSystem) delete theOSystem;
        if (game_controller) delete game_controller;
    }

    // Loads and initializes a game. After this call the game should be ready to play.
    bool loadROM(string rom_file, bool display_screen) {
        display_active = display_screen;
        int argc = 6;
        char** argv = new char*[argc];
        for (int i=0; i<=argc; i++) {
            argv[i] = new char[200];
        }
        strcpy(argv[0],"./ale");
        strcpy(argv[1],"-player_agent");
        strcpy(argv[2],"random_agent");
        strcpy(argv[3],"-display_screen");
        if (display_screen) strcpy(argv[4],"true");
        else                strcpy(argv[4],"false");
        strcpy(argv[5],rom_file.c_str());  

        cout << welcomeMessage() << endl;
    
        if (theOSystem) delete theOSystem;

#ifdef WIN32
        theOSystem = new OSystemWin32();
        SettingsWin32 settings(theOSystem);
#else
        theOSystem = new OSystemUNIX();
        SettingsUNIX settings(theOSystem);
#endif

        setDefaultSettings(theOSystem->settings());

        theOSystem->settings().loadConfig();

        // process commandline arguments, which over-ride all possible config file settings
        string romfile = theOSystem->settings().loadCommandLine(argc, argv);

        // Load the configuration from a config file (passed on the command
        //  line), if provided
        string configFile = theOSystem->settings().getString("config", false);
   
        if (!configFile.empty())
            theOSystem->settings().loadConfig(configFile.c_str());

        theOSystem->settings().validate();
        theOSystem->create();
  
        //// Main loop ////
        // First we check if a ROM is specified on the commandline.  If so, and if
        //   the ROM actually exists, use it to create a new console.
        if(argc == 1 || romfile == "" || !FilesystemNode::fileExists(romfile)) {
            printf("No ROM File specified or the ROM file was not found.\n");
            return false;
        } else if(theOSystem->createConsole(romfile)) 	{
            printf("Running ROM file...\n");
            theOSystem->settings().setString("rom_file", romfile);
        } else {
            printf("Unable to create console from ROM file.\n");
            return false;
        }

        // Seed the Random number generator
        if (theOSystem->settings().getString("random_seed") == "time") {
            cout << "Random Seed: Time" << endl;
            srand((unsigned)time(0)); 
        } else {
            int seed = theOSystem->settings().getInt("random_seed");
            assert(seed >= 0);
            cout << "Random Seed: " << seed << endl;
            srand((unsigned)seed); 
        }

        // Generate the GameController
        if (game_controller) delete game_controller;
        game_controller = new InternalController(theOSystem);
        theOSystem->setGameController(game_controller);

        // Set the palette 
        theOSystem->console().setPalette("standard");

        // Setup the screen representation
        mediasrc = &theOSystem->console().mediaSource();
        screen_width = mediasrc->width();
        screen_height = mediasrc->height();
        for (int i=0; i<screen_height; ++i) { // Initialize our screen matrix
            IntVect row;
            for (int j=0; j<screen_width; ++j)
                row.push_back(-1);
            screen_matrix.push_back(row);
        }

        // Intialize the ram array
        for (int i=0; i<RAM_LENGTH; i++)
            ram_content.push_back(0);

        emulator_system = &theOSystem->console().system();
        game_settings = buildRomRLWrapper(theOSystem->romFile());
        legal_actions = game_settings->getAllActions();
        minimal_actions = game_settings->getMinimalActionSet();
        max_num_frames = theOSystem->settings().getInt("max_num_frames", true);
    
        reset_game();

        return true;
    }

    // Resets the game
    void reset_game() {
        game_controller->systemReset();
       
        game_settings->reset();
        game_settings->step(*emulator_system);
        
        // Get the first screen
        mediasrc->update();
        int ind_i, ind_j;
        uInt8* pi_curr_frame_buffer = mediasrc->currentFrameBuffer();
        for (int i = 0; i < screen_width * screen_height; i++) {
            uInt8 v = pi_curr_frame_buffer[i];
            ind_i = i / screen_width;
            ind_j = i - (ind_i * screen_width);
            screen_matrix[ind_i][ind_j] = v;
        }

        // Get the first ram content
        for(int i = 0; i<RAM_LENGTH; i++) {
            int offset = i;
            offset &= 0x7f; // there are only 128 bytes
            ram_content[i] = emulator_system->peek(offset + 0x80);
        }

        // Record the starting time of this game
        time_start = time(NULL);
    }

    // Indicates if the game has ended
    bool game_over() {
        return game_settings->isTerminal() || (max_num_frames > 0 && frame > max_num_frames);
    }

    // Applies an action to the game and returns the reward. It is the user's responsibility
    // to check if the game has ended and reset when necessary -- this method will keep pressing
    // buttons on the game over screen.
    float act(Action action) {
        frame++;
        float action_reward = 0;
            
        // Apply action to simulator and update the simulator
        game_controller->getState()->apply_action(action, PLAYER_B_NOOP);

        // Get the latest screen
        mediasrc->update();
        int ind_i, ind_j;
        uInt8* pi_curr_frame_buffer = mediasrc->currentFrameBuffer();
        for (int i = 0; i < screen_width * screen_height; i++) {
            uInt8 v = pi_curr_frame_buffer[i];
            ind_i = i / screen_width;
            ind_j = i - (ind_i * screen_width);
            screen_matrix[ind_i][ind_j] = v;
        }

        // Get the latest ram content
        for(int i = 0; i<RAM_LENGTH; i++) {
            int offset = i;
            offset &= 0x7f; // there are only 128 bytes
            ram_content[i] = emulator_system->peek(offset + 0x80);
        }

        // Get the reward
        game_settings->step(*emulator_system);
        action_reward += game_settings->getReward();

        if (frame % 1000 == 0) {
            time_end = time(NULL);
            double avg = ((double)frame)/(time_end - time_start);
            cout << "Average main loop iterations per sec = " << avg << endl;
        }

        // Display the screen
        if (display_active) {
            theOSystem->p_display_screen->display_screen(screen_matrix, screen_width, screen_height);
            //theOSystem->p_display_screen->display_screen(*mediasrc);            
        }

        game_score += action_reward;
        last_action = action;
        return action_reward;
    }

};

#endif
