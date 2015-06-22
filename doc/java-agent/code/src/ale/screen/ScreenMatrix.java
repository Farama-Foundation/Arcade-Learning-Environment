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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;

/** Encapsulates screen matrix data. Also provides basic save/load operations on
 *   screen data.
 *
 * @author Marc G. Bellemare <mgbellemare@ualberta.ca>
 */
public class ScreenMatrix implements Cloneable {
    public int[][] matrix;
    public int width;
    public int height;

    /** Create a new, blank screen matrix with the given dimensions.
     * 
     * @param w width
     * @param h height
     */
    public ScreenMatrix(int w, int h) {
        matrix = new int[w][h];
        width = w;
        height = h;
    }

    /** Load a screen from a text file, in ALE format. The first line contains
     *   <width>,<height> .
     *  Each subsequent line (210 of them) contains a screen row with comma-separated
     *   values.
     * 
     * @param filename
     */
    public ScreenMatrix(String filename) throws IOException {
        // Create a BufferedReader to read in the data
        BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));

        // Obtain the width and height
        String line = in.readLine();
        String[] tokens = line.split(",");

        width = Integer.parseInt(tokens[0]);
        height = Integer.parseInt(tokens[1]);

        this.matrix = new int[width][height];

        int rowIndex = 0;

        // Read in the screen row-by-row, each separated by a newline
        while ((line = in.readLine()) != null) {
            // A row is a comma-separated list of integer values
            tokens = line.split(",");
            assert (tokens.length == width);

            for (int x = 0; x < tokens.length; x++) {
                this.matrix[x][rowIndex] = Integer.parseInt(tokens[x]);
            }

            rowIndex++;
        }
    }

    /** Saves this screen matrix as a text file. Can then be loaded using the
     *   relevant constructor.
     * 
     * @param filename
     * @throws IOException
     */
    public void saveData(String filename) throws IOException {
        PrintStream out = new PrintStream(new FileOutputStream(filename));

        // Width,height\n
        out.println(width+","+height);

        // Print the matrix, one row per line
        for (int y = 0; y < height; y++) {
            // Data is comma separated
            for (int x = 0; x < width; x++) {
                out.print(matrix[x][y]);
                if (x < width - 1) out.print(",");
            }

            out.println();
        }
    }

    /** Clones this screen matrix. Data is copied.
     * 
     * @return
     */
    @Override
    public Object clone() {
        try {
            ScreenMatrix img = (ScreenMatrix)super.clone();

            // Create a new matrix which we will fill with the proper data
            img.matrix = new int[this.width][this.height];
        
            for (int x = 0; x < this.width; x++) {
                System.arraycopy(this.matrix[x], 0, img.matrix[x], 0, this.height);
            }
            return img;
        }
        catch (CloneNotSupportedException e) {
            return null;
        }
    }
}
