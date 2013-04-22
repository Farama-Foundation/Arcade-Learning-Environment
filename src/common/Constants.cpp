/* *****************************************************************************
 * A.L.E (Atari 2600 Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf
 * Released under GNU General Public License www.gnu.org/licenses/gpl-3.0.txt
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  Constants.cpp
 *
 *  Defines a set of constants used by various parts of the player agent code
 *
 **************************************************************************** */

#include "Constants.h"

std::string action_to_string(Action a) {
    static string tmp_action_to_string[] = {
        "PLAYER_A_NOOP"          
        ,"PLAYER_A_FIRE"          
        ,"PLAYER_A_UP"            
        ,"PLAYER_A_RIGHT"         
        ,"PLAYER_A_LEFT"          
        ,"PLAYER_A_DOWN"          
        ,"PLAYER_A_UPRIGHT"       
        ,"PLAYER_A_UPLEFT"        
        ,"PLAYER_A_DOWNRIGHT"     
        ,"PLAYER_A_DOWNLEFT"      
        ,"PLAYER_A_UPFIRE"        
        ,"PLAYER_A_RIGHTFIRE"     
        ,"PLAYER_A_LEFTFIRE"      
        ,"PLAYER_A_DOWNFIRE"      
        ,"PLAYER_A_UPRIGHTFIRE"   
        ,"PLAYER_A_UPLEFTFIRE"    
        ,"PLAYER_A_DOWNRIGHTFIRE"
        ,"PLAYER_A_DOWNLEFTFIRE"
        ,"PLAYER_B_NOOP"          
        ,"PLAYER_B_FIRE"          
        ,"PLAYER_B_UP"            
        ,"PLAYER_B_RIGHT"         
        ,"PLAYER_B_LEFT"          
        ,"PLAYER_B_DOWN"          
        ,"PLAYER_B_UPRIGHT"       
        ,"PLAYER_B_UPLEFT"        
        ,"PLAYER_B_DOWNRIGHT"     
        ,"PLAYER_B_DOWNLEFT"      
        ,"PLAYER_B_UPFIRE"        
        ,"PLAYER_B_RIGHTFIRE"     
        ,"PLAYER_B_LEFTFIRE"      
        ,"PLAYER_B_DOWNFIRE"      
        ,"PLAYER_B_UPRIGHTFIRE"   
        ,"PLAYER_B_UPLEFTFIRE"    
        ,"PLAYER_B_DOWNRIGHTFIRE"
        ,"PLAYER_B_DOWNLEFTFIRE"
        ,"__invalid__" // 36
        ,"__invalid__" // 37
        ,"__invalid__" // 38
        ,"__invalid__" // 39
        ,"RESET"       // 40
        ,"UNDEFINED"   // 41
        ,"RANDOM"      // 42
    };
    assert (a >= 0 && a <= 42);
    return tmp_action_to_string[a];
}
