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
 *  ScreenSDL.cpp
 *
 *  Supports displaying the screen via SDL.
 **************************************************************************** */
#ifdef SDL_SUPPORT
#include "ale/common/SDL2.hpp"

#include "ale/common/ScreenSDL.hpp"
#include "ale/common/Log.hpp"

#include <cmath>

using namespace std;

namespace ale {
using namespace stella;

ScreenSDL::ScreenSDL(OSystem* osystem) :
  Screen(osystem),
  mediaSource(NULL),
  sound(NULL),
  colourPalette(NULL)
{
  // Store necesarry info for ScreenSDL
  mediaSource = &osystem->console().mediaSource();
  sound = &osystem->sound();
  colourPalette = &osystem->colourPalette();
  maxFPS = FPS = osystem->console().getFrameRate();

  screenHeight = mediaSource->height();
  screenWidth = mediaSource->width();

  // Initialize SDL video
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    throw std::runtime_error("Failed to initialize SDL");
  }

  // We use the standard ARGB8888 pixel format.
  pixelFormat = SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);

  // Get an appropriate scale factor for our screen resolution.
  int scaleFactor = getScaleFactor();

  // Create the ALE window scaling up by `scaleFactor`.
  window = SDL_CreateWindow("The Arcade Learning Environment",
                SDL_WINDOWPOS_UNDEFINED,
                SDL_WINDOWPOS_UNDEFINED,
                windowWidth * scaleFactor,
                windowHeight * scaleFactor,
                SDL_WINDOW_ALLOW_HIGHDPI);
  if (window == nullptr) {
    throw std::runtime_error("Failed to initialize SDL window");
  }


  // Get the render info for our driver.
  SDL_RendererInfo rendererInfo;
  if (SDL_GetRenderDriverInfo(0, &rendererInfo) < 0) {
    throw std::runtime_error("Failed to query renderer 0");
  }
  // If we're using the dummy driver (as in test cases) make sure
  // we have no render flags set
  if(strcmp(SDL_GetCurrentVideoDriver(), "dummy") == 0) {
    rendererInfo.flags = 0;
  } else {
    rendererInfo.flags |= SDL_RENDERER_PRESENTVSYNC;
  }

  // Create our renderer
  renderer = SDL_CreateRenderer(window, -1, rendererInfo.flags);
  if (renderer == nullptr) {
    throw std::runtime_error("Failed to initialize SDL renderer");
  }
  // Make sure we set the logical size and allow for integer scaling so
  // our scaling factor acts as intended
  SDL_RenderSetLogicalSize(renderer, windowWidth, windowHeight);
  SDL_RenderSetIntegerScale(renderer, SDL_TRUE);

  // Create our texture with streaming access, we just blit the entire screen
  // to this texture.
  texture = SDL_CreateTexture(renderer, pixelFormat->format, SDL_TEXTUREACCESS_STREAMING, screenWidth, screenHeight);
  if (texture == nullptr) {
    throw std::runtime_error("Failed to initialize SDL texture");
  }

  // Create the SDL surface
  surface = SDL_CreateRGBSurface(0,
    windowWidth,
    windowHeight,
    pixelFormat->BitsPerPixel,
    pixelFormat->Rmask,
    pixelFormat->Gmask,
    pixelFormat->Bmask,
    pixelFormat->Amask);
  if (surface == nullptr) {
    throw std::runtime_error("Failed to initialize SDL surface");
  }

  // Keep track of the last render time so we can match
  // our target FPS.
  lastRender = SDL_GetTicks();
}

ScreenSDL::~ScreenSDL() {
  // Make sure to cleanup SDL primitives
  SDL_FreeSurface(surface);
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

void ScreenSDL::render() {
  // Get output buffer from surface, current frame buffer from media source
  // and compute the proper pitch.
  uint32_t* out = reinterpret_cast<uint32_t*>(surface->pixels);
  uint8_t* buffer = mediaSource->currentFrameBuffer();
  uint32_t pitch = surface->pitch / pixelFormat->BytesPerPixel;

  // Copy over the screen data to the surface
  uint32_t bufferOffset = 0, offsetY = 0, pos;
  for (uint32_t y = 0; y < screenHeight; ++y) {
    pos = offsetY;
    for(uint32_t x = screenWidth / 2; x; --x) {
      out[pos++] = colourPalette->getRGB(buffer[bufferOffset++]);
      out[pos++] = colourPalette->getRGB(buffer[bufferOffset++]);
    }
    offsetY += pitch;
  }

  // Update the texture and present the result to renderer
  SDL_UpdateTexture(texture, nullptr, surface->pixels, surface->pitch);
  SDL_RenderCopy(renderer, texture, nullptr, nullptr);
  SDL_RenderPresent(renderer);

  // Poll for events, the window will become unresponsive if we
  // don't poll, for say, window manager events
  poll();

  // Try and maintain our target FPS by delaying the appropriate
  // amount to average FPS.
  uint32_t thisTick = SDL_GetTicks();
  uint32_t delta = thisTick - min(lastRender, thisTick);
  uint32_t renderDelay = 1000 / FPS;
  if (delta < renderDelay) {
    SDL_Delay(renderDelay - delta);
  } else {
    lastRender = thisTick + delta - renderDelay;
  }
}

void ScreenSDL::poll() {
  // Poll for SDL events
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    handleSDLEvent(event);
  }
}

int ScreenSDL::getScaleFactor() {
  // Try to get the maximum integer scaling factor
  // that won't exceed 40% of max(height, width)
  // of the users current display.
  SDL_DisplayMode mode;
  SDL_GetCurrentDisplayMode(0, &mode);

  float maxScaleProportion = 0.4;

  int widthHeuristic = ceil(mode.w * maxScaleProportion);
  int heightHeuristic = ceil(mode.h * maxScaleProportion);
  int scaleFactor;

  if (widthHeuristic > heightHeuristic) {
    scaleFactor = round(
        static_cast<float>(widthHeuristic) / static_cast<float>(windowWidth));
  } else {
    scaleFactor = round(
        static_cast<float>(heightHeuristic) / static_cast<float>(windowHeight));
  }

  // Make sure we have a scaling factor of at least 1
  return max(scaleFactor, 1);
}

void ScreenSDL::handleSDLEvent(const SDL_Event& event) {
  // TODO: OSD for keystrokes
  switch(event.type) {
    case SDL_QUIT:
      std::exit(0);
      break;
    case SDL_KEYDOWN:
      switch(event.key.keysym.sym) {
        case SDLK_LEFT:
          FPS = max(FPS - 5, 5u);
          sound->setFrameRate(FPS);
          break;
        case SDLK_RIGHT:
          FPS = min(maxFPS, FPS + 5);
          sound->setFrameRate(FPS);
          break;
        case SDLK_DOWN:
          for (int i = 0; i < 5; i++)
            sound->adjustVolume(-1);
          break;
        case SDLK_UP:
          for (int i = 0; i < 5; i++)
            sound->adjustVolume(1);
          break;
      }
  }
}

} //namespace ale

#endif // SDL_SUPPORT
