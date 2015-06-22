/*
 * Java Arcade Learning Environment (A.L.E) Agent
 *  Copyright (C) 2011-2012 Marc G. Bellemare <mgbellemare@ualberta.ca>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package ale.io;

import java.util.HashMap;

/** A static container for Atari actions.
 *
 * @author Marc G. Bellemare
 */
public class Actions {
    /** The number of player actions available to each player */
    public static int numPlayerActions = 18;

    /** A list of all the action names */
    public static String[] actionNames = {
        "player_a_noop",
        "player_a_fire",
        "player_a_up",
        "player_a_right",
        "player_a_left",
        "player_a_down",
        "player_a_upright",
        "player_a_upleft",
        "player_a_downright",
        "player_a_downleft",
        "player_a_upfire",
        "player_a_rightfire",
        "player_a_leftfire",
        "player_a_downfire",
        "player_a_uprightfire",
        "player_a_upleftfire",
        "player_a_downrightfire",
        "player_a_downleftfire",
        "player_b_noop",
        "player_b_fire",
        "player_b_up",
        "player_b_right",
        "player_b_left",
        "player_b_down",
        "player_b_upright",
        "player_b_upleft",
        "player_b_downright",
        "player_b_downleft",
        "player_b_upfire",
        "player_b_rightfire",
        "player_b_leftfire",
        "player_b_downfire",
        "player_b_uprightfire",
        "player_b_upleftfire",
        "player_b_downrightfire",
        "player_b_downleftfire",
        "reset",
        "undefined",
        "random",
        // MGB v0.2 actions
        "save_state",
        "load_state",
        "system_reset"
    };

    /** A HashMap mapping action names to action indices */
    public static HashMap<String,Integer> actionsMap;

    /** Maps a given action name to its corresponding integer value */
    public static int map(String actionName) {
        if (actionsMap == null) makeMap();

        return actionsMap.get(actionName).intValue();
    }

    /** Construct the map from names to actions */
    public static void makeMap() {
        actionsMap = new HashMap<String,Integer>();
        
        for (int i = 0; i < actionNames.length; i++) {
            int v;

            if (i < numPlayerActions * 2) v = i;
            // Special actions (not player-related) start at 40
            else {
                v = i + 4;
            }
            actionsMap.put(actionNames[i], new Integer(v));
        }
    }
}
