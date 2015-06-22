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
package ale.gui;

import ale.io.Actions;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

/** A crude keyboard controller. The following keys are mapped:
 *    R - reset
 *    Q - quit
 *    space - fire
 *    ASWD, arrow keys - joystick movement
 *
 * @author Marc G. Bellemare
 */
public class KeyboardControl implements KeyListener {
    /** Variables used to keep track of which keys are pressed */
    public boolean up, down;
    public boolean left, right;
    public boolean fire;

    public boolean reset;

    public boolean quit;

    /** Creates a new keyboard controller.
     * 
     */
    public KeyboardControl() {
        // Initially all keys are assumed not pressed
        up = down = left = right = fire = false;
        reset = false;
        quit = false;
    }

    public void keyTyped(KeyEvent e) {
    }

    public void keyPressed(KeyEvent e) {
        // Parse different key presses by setting the relevant boolean flags
        switch (e.getKeyCode()) {
          case KeyEvent.VK_UP:
          case KeyEvent.VK_W:
            up = true;
            break;
          case KeyEvent.VK_DOWN:
          case KeyEvent.VK_S:
            down = true;
            break;
          case KeyEvent.VK_LEFT:
          case KeyEvent.VK_A:
            left = true;
            break;
          case KeyEvent.VK_RIGHT:
          case KeyEvent.VK_D:
            right = true;
            break;
          case KeyEvent.VK_SPACE:
            fire = true;
            break;
          case KeyEvent.VK_R:
            reset = true;
            break;
          case KeyEvent.VK_ESCAPE:
            quit = true;
            break;
        }
    }

    public void keyReleased(KeyEvent e) {
        // Opposite of keyPressed; sets the relevant boolean flag to false
        switch (e.getKeyCode()) {
          case KeyEvent.VK_UP:
          case KeyEvent.VK_W:
            up = false;
            break;
          case KeyEvent.VK_DOWN:
          case KeyEvent.VK_S:
            down = false;
            break;
          case KeyEvent.VK_LEFT:
          case KeyEvent.VK_A:
            left = false;
            break;
          case KeyEvent.VK_RIGHT:
          case KeyEvent.VK_D:
            right = false;
            break;
          case KeyEvent.VK_SPACE:
            fire = false;
            break;
          case KeyEvent.VK_R:
            reset = false;
            break;
          case KeyEvent.VK_ESCAPE:
            quit = false;
            break;
        }
    }

    /** An array to map a bit-wise representation of the keypresses to ALE actions.
      * 1 = fire, 2 = up, 4 = right, 8 = left, 16 = down
      *
      * -1 indicate an invalid combination, e.g. left/right or up/down. These should
      * be filtered out in toALEAction.
      */
    private int[] bitKeysMap = new int[] {
        0, 1, 2, 10, 3, 11, 6, 14, 4, 12, 7, 15, -1, -1, -1, -1,
        5, 13, -1, -1, 8, 16, -1, -1, 9, 17, -1, -1, -1, -1, -1, -1
    };
    
    /** Converts the current keypresses to an ALE action (for player A).
     * 
     * @return
     */
    public int toALEAction() {
        int bitfield = 0;

        // Reset overrides everything
        if (reset) return Actions.map("system_reset");

        // Cancel out left/right, up/down; obtain the corresponding bit representation
        if (left == right) bitfield |= 0;
        else if (left) bitfield |= 0x08;
        else if (right) bitfield |= 0x04;

        if (up == down) bitfield |= 0;
        else if (up) bitfield |= 0x02;
        else if (down) bitfield |= 0x10;

        if (fire) bitfield |= 0x01;

        // Map the bits to an ALE action
        return bitKeysMap[bitfield];
    }

}
