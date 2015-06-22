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

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import javax.swing.JPanel;

/** Displays the current Atari frame in a simple GUI.
 *
 * @author Marc G. Bellemare
 */
public class ScreenDisplay extends JPanel {

    /** The image to be displayed */
    BufferedImage image;
    /** The scale at which we want to display (3x normal height) */
    int yScaleFactor = 3;
    /** The x-axis scale at which we want to display (6x normal width) */
    int xScaleFactor = 6;
    /** The default screen width */
    int defaultWidth = 160;
    /** The default screen height */
    int defaultHeight = 210;
    /** The height of the status bar at the bottom of the GUI */
    int statusBarHeight = 20;
    /** Variables storing some relevant GUI dimensions */
    int statusBarY;
    int windowWidth;
    int windowHeight;
    /** Variables used to compute the GUI frames per second */
    int frameCount = 0;
    double fps = 0;
    long frameTime = 0;
    int updateRate = 5; // How often to update FPS, in hertz
    double fpsAlpha = 0.9;
    
    /** Additional user strings to be displayed */
    String centerString;
    MessageHistory messages;

    long maxMessageAge = 3000;
    
    public ScreenDisplay() {
        super();

        messages = new MessageHistory();
    }

    public Dimension getPreferredSize() {
        int width, height;

        statusBarY = defaultHeight * yScaleFactor;
        width = defaultWidth * xScaleFactor;
        height = statusBarY + statusBarHeight;

        windowWidth = width;
        windowHeight = height;

        return new Dimension(width, height);
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        drawImages(g);
    }

    public void setImage(BufferedImage img) {
        synchronized (this) {
            this.image = img;
        }
    }

    public void setCenterString(String s) {
        synchronized (this) {
            centerString = s;
        }
    }

    public void addMessage(String s) {
        synchronized (this) {
            messages.addMessage(s);
        }
    }

    /** This methods calculates how many frames per second are being displayed.
     *   Exponential averaging is used for smoothness.
     */
    public void updateFrameCount() {
        synchronized (this) {
            frameCount++;
            long time = System.currentTimeMillis();

            // If one second has elapsed, update FPS
            if (time - frameTime >= 1000 / updateRate) {
                if (fps == 0) {
                    fps = frameCount;
                } else {
                    // Compute the exact number of (fractional) ticks since FPS update
                    double ticksSinceUpdate = (time - frameTime) * updateRate / 1000.0;
                    double alpha = Math.pow(fpsAlpha, ticksSinceUpdate);

                    fps = alpha * fps + (1 - alpha) * (frameCount * updateRate / ticksSinceUpdate);
                }

                frameCount = 0;
                frameTime = time;
            }
        }
    }

    /** Helper method that the display by the given (x,y) factors.
     *
     * @param g
     * @param xFactor 
     * @param yFactor
     */
    private void rescale(Graphics g, double xFactor, double yFactor) {
        if (g instanceof Graphics2D) {
            Graphics2D g2d = (Graphics2D) g;
            g2d.scale(xFactor, yFactor);
        }
    }

    public void drawImages(Graphics g) {
        synchronized (this) {
            // Do some message cleanup if necessary
            messages.update(maxMessageAge);

            // Zoom up on the Atari image
            rescale(g, xScaleFactor, yScaleFactor);
            // draw the atari image
            if (image != null) {
                g.drawImage(image, 0, 0, null);
            }

            // Zoom out to draw text
            rescale(g, 1.0 / xScaleFactor, 1.0 / yScaleFactor);

            int statusBarTextOffset = statusBarY + 15;

            // draw FPS information in the bottom left corner
            if (fps > 0) {
                g.setColor(Color.BLACK);
                double roundedFPS = (Math.round(fps * 10) / 10.0);
                g.drawString("FPS: " + roundedFPS, 0, statusBarTextOffset);
            }

            // Draw a string center-bottom
            if (centerString != null) {
                int stringLength = g.getFontMetrics().stringWidth(centerString);
                g.drawString(centerString, (windowWidth - stringLength) / 2, statusBarTextOffset);
            }

            int textOffset = statusBarY - 4;

            g.setColor(Color.YELLOW);

            // Draw messages in the bottom right corner
            for (MessageHistory.Message m : messages.getMessages()) {
                // Draw one message
                String text = m.getText();
                int stringLength = g.getFontMetrics().stringWidth(text);
                g.drawString(text, windowWidth - stringLength - 2, textOffset);

                // Decrement textOffset so that the next (older) message
                //  is drawn on top of it
                textOffset -= g.getFontMetrics().getHeight();
            }
        }
    }
}
