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
package ale.rl;

import ale.screen.ScreenMatrix;
import java.util.LinkedList;

/** A time-ordered list of frames.
 *
 * @author Marc G. Bellemare
 */
public class FrameHistory implements Cloneable {
    /** The list of recent frames */
    protected LinkedList<ScreenMatrix> frames;

    /** The maximum length of history we need to keep */
    protected int maxLength;

    /** Create a new FrameHistory which needs to keep no more than the last
     *    'maxLength' frames.
     * 
     * @param maxLength
     */
    public FrameHistory(int maxLength) {
        this.maxLength = maxLength;
        frames = new LinkedList<ScreenMatrix>();
    }

    /** Append a new frame to the end of the history.
     * 
     * @param frame
     */
    public void addFrame(ScreenMatrix frame) {
        frames.addLast(frame);
        while (frames.size() > maxLength)
            frames.removeFirst();
    }

    /** Removes the t-to-last frame. For example, removeLast(0) removes the 
     *   last frame added by addFrame(frame).
     */
    public void removeLast(int t) {
        frames.remove(frames.size() - t - 1);
    }

    public int maxHistoryLength() {
        return maxLength;
    }

    /** Returns the t-to-last frame. For example, getLastFrame(0) returns the
     *   last frame added by addFrame(frame).
     */
    public ScreenMatrix getLastFrame(int t) {
        return frames.get(frames.size() - t - 1);
    }

    public Object clone() {
        try {
            FrameHistory obj = (FrameHistory)super.clone();

            obj.frames = new LinkedList<ScreenMatrix>();
            // Copy over the frames; we do not clone them
            for (ScreenMatrix screen : this.frames) {
                obj.frames.add(screen);
            }
            return obj;
        }
        catch (CloneNotSupportedException e) {
            return null;
        }
    }
}
