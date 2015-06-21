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

/** This class defines the SECAM color palette. The entries in this class were
 *  taken from Stella.
 *
 * @author Marc G. Bellemare <mgbellemare@ualberta.ca>
 */
public class SECAMPalette extends ColorPalette {
    // 8 unique colors, with odd indices being mapped to the color below

    protected int[] colorData = new int[]{
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
        0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
        0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0
    };

    /** Creates a new SECAM palette.
     * 
     */
    public SECAMPalette() {
        super();

        // Set the palette as given above
        for (int index = 0; index < colorData.length; index++) {
            int v = colorData[index & ~0x1];
            int r = (v & 0xFF0000) >> 16;
            int g = (v & 0x00FF00) >> 8;
            int b = v & 0x0000FF;

            super.set(new Color(r, g, b), index);
        }
    }
}
