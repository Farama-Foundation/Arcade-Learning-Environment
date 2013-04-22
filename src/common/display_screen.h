/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare, 
 *   Matthew Hausknecht, and the Reinforcement Learning and Artificial Intelligence 
 *   Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  diplay_screen.cpp 
 *
 *  Supports displaying the screen via SDL. 
 **************************************************************************** */

#ifndef DISPLAY_SCREEN_H
#define DISPLAY_SCREEN_H

#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "export_screen.h"
#include "../emucore/MediaSrc.hxx"

#ifdef __USE_SDL
#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "SDL/SDL_rotozoom.h"
#include "SDL/SDL_gfxPrimitives.h"

class SDLEventHandler {
public:
    // Returns true if it handles this SDL Event. False if not and the event
    // will be passed to other handlers.
    virtual bool handleSDLEvent(const SDL_Event& event) = 0;

    // This gives the handler a chance to draw on or modify the screen that
    // will be displayed. This is done by modifying the screen_matrix.
    virtual void display_screen(IntMatrix& screen_matrix, int screen_width, int screen_height) = 0;

    // Print the usage information about this handler
    virtual void usage() = 0;
};

class DisplayScreen : public SDLEventHandler {
public:
    DisplayScreen(ExportScreen* export_screen, int screen_width, int screen_height);
    virtual ~DisplayScreen();

    // Displays the current frame buffer directly from the mediasource
    void display_screen(const MediaSource& mediaSrc);

    // Displays a screen_matrix. This is called after all other handlers
    // draw on the screen.
    void display_screen(IntMatrix& screen_matrix, int image_width, int image_height);

    // Draws a png image to the screen from a file
    void display_png(const string& filename);

    // Registers a handler for keyboard and mouse events
    void registerEventHandler(SDLEventHandler* handler);

    // Allows other methods to set the paused/unpaused game status
    void setPaused(bool _paused) { paused = _paused; }

    // Implements pause functionality
    bool handleSDLEvent(const SDL_Event& event);

    void usage();


public:
    // Dimensions of the SDL window
    int window_height, window_width;

    // Maintains the paused/unpaused state of the game
    bool paused;


protected:
    // Checks for SDL events such as keypresses.
    // TODO: Run in a different thread?
    void poll();

    SDL_Surface *screen, *image;
    ExportScreen* export_screen;

    // Matrix representation of the screen
    IntMatrix screen_matrix;
    int screen_height, screen_width;

    // Handlers for SDL Events
    std::vector<SDLEventHandler*> handlers;
};
#else
/** A dummy class that simply ignores display events. */
class DisplayScreen {
  public:
    DisplayScreen(ExportScreen* export_screen, int screen_width, int screen_height) {}

    // Displays the current frame buffer directly from the mediasource
    void display_screen(const MediaSource& mediaSrc) {}

    // Displays a screen_matrix. This is called after all other handlers
    // draw on the screen.
    void display_screen(IntMatrix& screen_matrix, int image_width, int image_height) {}
};
#endif // __USE_SDL

#endif // DISPLAY_SCREEN
