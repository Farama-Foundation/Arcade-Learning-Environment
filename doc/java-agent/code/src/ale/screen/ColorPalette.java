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
package ale.screen;

import java.awt.Color;

/** Defines a palette of colors. Up to 256 entries. 0 is always black.
 *
 * @author Marc G. Bellemare <mgbellemare@ualberta.ca>
 */
public abstract class ColorPalette {
    /** 256 colors in this palette */
    public static final int MAX_ENTRIES = 256;

    /** A map of screen indices to RGB colors. */
    protected Color[] map;
    /** How many entries our map contains. */
    protected int numEntries;

    /** Create a new map, with entry #0 being black.
     * 
     */
    public ColorPalette() {
        map = new Color[MAX_ENTRIES];
        // 0 is always black
        set(Color.BLACK, 0);
    }

    /** Returns how many entries are contained in this color map.
     * 
     * @return
     */
    public int numEntries() {
        return this.numEntries;
    }

    /** Adds Color c at index i.
     *
     * @param c Color
     * @param i index
     */
    public Color set(Color c, int i) {
        Color oldColor = map[i];

        map[i] = c;
        if (oldColor == null) numEntries++;

        return oldColor;
    }

    /** Returns the color indexed by i, possibly null.
     * 
     * @param i
     * @return
     */
    public Color get(int i) {
        return map[i];
    }
    
    /** Returns whether palette index i has an associated color.
     * 
     * @param i
     * @return
     */
    public boolean hasEntry(int i) {
        return (map[i] != null);
    }    
}
