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
#include "display_screen.h"

#ifdef __USE_SDL
DisplayScreen::DisplayScreen(ExportScreen* _export_screen, int _screen_width, int _screen_height):
    window_height(420), window_width(337), paused(false), export_screen(_export_screen),
    screen_height(_screen_height), screen_width(_screen_width)
{
    // Initialise SDL Video */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }

    screen = SDL_SetVideoMode(window_width,window_height, 0, SDL_HWPALETTE|SDL_DOUBLEBUF|SDL_RESIZABLE);
	
    if (screen == NULL) {
        fprintf(stderr, "Couldn't Initialize Screen: %s\n", SDL_GetError());
        exit(-1);
    }

    // Set the screen title
    SDL_WM_SetCaption("A.L.E. Viz", NULL);

    // Initialize our screen matrix
    for (int i=0; i<screen_height; ++i) { 
        IntVect row;
        for (int j=0; j<screen_width; ++j)
            row.push_back(-1);
        screen_matrix.push_back(row);
    }

    // Register ourselves as an event handler
    registerEventHandler(this);

    fprintf(stderr, "Screen Display Active. Press 'h' for help.\n");
}

DisplayScreen::~DisplayScreen() {
    // Shut down SDL 
    SDL_Quit();
}

void DisplayScreen::display_screen(const MediaSource& mediaSrc) {
    // Convert the media sources frame into the screen matrix representation
    uInt8* pi_curr_frame_buffer = mediaSrc.currentFrameBuffer();
    int ind_i, ind_j;
    for (int i = 0; i < screen_width * screen_height; i++) {
        uInt8 v = pi_curr_frame_buffer[i];
        ind_i = i / screen_width;
        ind_j = i - (ind_i * screen_width);
        screen_matrix[ind_i][ind_j] = v;
    }

    // Give our handlers a chance to mess with the screen
    for (int i=handlers.size()-1; i>=0; --i) {
        handlers[i]->display_screen(screen_matrix, screen_width, screen_height);
    }
}

void DisplayScreen::display_screen(IntMatrix& screen_matrix, int screen_width, int screen_height) {
    Uint32 rmask, gmask, bmask, amask;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    rmask = 0xff000000;
    gmask = 0x00ff0000;
    bmask = 0x0000ff00;
    amask = 0x000000ff;
#else
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0xff000000;
#endif

    SDL_Surface* my_surface = SDL_CreateRGBSurface(SDL_SWSURFACE,screen_width,screen_height,32,rmask,gmask,bmask,amask);

    int r, g, b;
    for (int y=0; y<screen_height; ++y) {
        for (int x=0; x<screen_width; ++x) {
            export_screen->get_rgb_from_palette(screen_matrix[y][x], r, g, b);
            pixelRGBA(my_surface,x,y,r,g,b,255);
        }
    }

    SDL_Surface* zoomed = zoomSurface(my_surface,screen->w/(double)screen_width,screen->h/(double)screen_height,0);
    SDL_BlitSurface(zoomed, NULL, screen, NULL);

    SDL_Flip(screen);
    SDL_FreeSurface(my_surface);
    SDL_FreeSurface(zoomed);
    poll(); // Check for quit event
}


void DisplayScreen::display_png(const string& filename) {
    image = IMG_Load(filename.c_str());
    if ( !image ) {
        fprintf (stderr, "IMG_Load: %s\n", IMG_GetError () );
    } 

    // Draws the image on the screen:
    SDL_Rect rcDest = { 0,0, (Uint16)(2*image->w), (Uint16)(2*image->h) };
    SDL_Surface *image2 = zoomSurface(image, 2.0, 2.0, 0);
    SDL_BlitSurface ( image2, NULL, screen, &rcDest );
    // something like SDL_UpdateRect(surface, x_pos, y_pos, image->w, image->h); is missing here

    SDL_Flip(screen);

    SDL_FreeSurface(image);
    SDL_FreeSurface(image2);
    SDL_FreeSurface(screen);
    poll(); // Check for quit event
}

void DisplayScreen::registerEventHandler(SDLEventHandler* handler) {
    handlers.push_back(handler);
};

void DisplayScreen::poll() {
    SDL_Event event;
    while(SDL_PollEvent(&event)) {
        switch (event.type) {
        case SDL_QUIT:
            exit(0);
            break;
        case SDL_VIDEORESIZE:
            screen = SDL_SetVideoMode(event.resize.w,event.resize.h, 0, SDL_HWPALETTE|SDL_DOUBLEBUF|SDL_RESIZABLE);
            break;
        default:
            break;
        }

        // Give our event handlers a chance to deal with this event.
        // Latest handlers get first chance.
        for (int i=handlers.size()-1; i>=0; --i) {
            if (handlers[i]->handleSDLEvent(event))
                break;
        }

        // Keep polling while paused. This looks fishy but works... o.O
        while(paused) {
            poll();
            SDL_Delay(10);
        }
    }
};

bool DisplayScreen::handleSDLEvent(const SDL_Event& event) {
    switch (event.type) {
    case SDL_KEYDOWN:
        switch(event.key.keysym.sym){
        case SDLK_SPACE:
            paused = !paused;
            if (paused) fprintf(stderr, "Paused...\n");
            else fprintf(stderr, "Unpaused...\n");
            return true;
        case SDLK_h: // Print help info
            fprintf(stderr, "Screen Display Commands:\n");
            for (uint i=0; i<handlers.size(); ++i) {
                handlers[i]->usage();
            }
            break;
        default:
            break;
        }
    }
    return false;
};

void DisplayScreen::usage() {
    fprintf(stderr, "  -h: print help info\n  -Space Bar: Pause/Unpause game.\n");
};
#endif
