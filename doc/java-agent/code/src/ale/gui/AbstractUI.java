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

import java.awt.image.BufferedImage;

/** An interface describing a UI. This gets subclassed into a graphical UI,
 *   a command-line UI, etc... as needed.
 *
 * @author Marc G. Bellemare
 */
public interface AbstractUI {
    /** This method is called to notify the UI that we want to terminate. */
    public void die();
    /** Notifies the UI that it should refresh its display */
    public void refresh();

    /** Sets the screen image to be displayed in the GUI */
    public void setImage(BufferedImage img);

    /** Provides a string to be displayed (at the bottom of the GUI if using a GUI) */
    public void setCenterString(String s);
    public void addMessage(String s);

    /** Obtain an ALE action from the UI, e.g. via the keyboard.
     * 
     * @return
     */
    public int getKeyboardAction();
    /** Returns true if the user requested the end of the program, e.g. via a
     *   keypress.
     * @return
     */
    public boolean quitRequested();

    /** A method called to notify the UI that a new frame has been processed.
     *   Used to display frames per second information.
     */
    public void updateFrameCount();
}
