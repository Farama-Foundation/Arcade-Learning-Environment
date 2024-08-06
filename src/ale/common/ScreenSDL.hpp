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
 *  ScreenSDL.hpp
 *
 *  Supports displaying the screen via SDL.
 **************************************************************************** */

#ifndef _SCREEN_SDL_HPP
#define _SCREEN_SDL_HPP

#ifdef SDL_SUPPORT

#include "ale/common/Constants.h"
#include "ale/common/ColourPalette.hpp"
#include "ale/emucore/Screen.hxx"
#include "ale/emucore/OSystem.hxx"
#include "ale/emucore/MediaSrc.hxx"
#include "ale/common/SDL2.hpp"

namespace ale {

class ScreenSDL : public stella::Screen {
public:
    ScreenSDL(stella::OSystem* osystem);
    virtual ~ScreenSDL();

    // Displays the current frame buffer from the mediasource.
    void render();
private:
    // Poll for SDL events.
    void poll();

    // Handle the SDL_Event.
    void handleSDLEvent(const SDL_Event& event);

    // Get scaling factor from screen resolution
    int getScaleFactor();
private:
    // 4:3 Aspect ratio default 2600 screen
    static const uint32_t windowHeight = 321;
    static const uint32_t windowWidth = 428;
    uint32_t screenWidth, screenHeight;

    // ALE primitives
    stella::MediaSource* mediaSource;
    stella::Sound* sound;
    ColourPalette* colourPalette;

    // SDL Primitives
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    SDL_Surface* surface;
    SDL_PixelFormat* pixelFormat;

    // Used to calibrate delay between frames
    uint32_t lastRender;
    uint32_t FPS, maxFPS;
};

} // namespace ale

#endif // SDL_SUPPORT

#endif // DISPLAY_SCREEN
