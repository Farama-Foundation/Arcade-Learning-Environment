/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  SDL2.cpp
 *
 *  SDL2 shim for functions in the ALE.
 **************************************************************************** */

/*
 * If we're dynamically loading we must shim
 * all the SDL functions to dynamically link
 * their function calls to SDL.
 *
 * We could have used a fancier macro but
 * manual definition is more clear.
 */
#ifdef SDL_DYNLOAD
#include "ale/common/SDL2.hpp"

namespace ale {

// Define SDL2 function pointers
DEFINE_SDL2_POINTER(SDL_Init);
DEFINE_SDL2_POINTER(SDL_AllocFormat);
DEFINE_SDL2_POINTER(SDL_CloseAudio);
DEFINE_SDL2_POINTER(SDL_CreateRGBSurface);
DEFINE_SDL2_POINTER(SDL_CreateRenderer);
DEFINE_SDL2_POINTER(SDL_CreateTexture);
DEFINE_SDL2_POINTER(SDL_CreateWindow);
DEFINE_SDL2_POINTER(SDL_Delay);
DEFINE_SDL2_POINTER(SDL_DestroyRenderer);
DEFINE_SDL2_POINTER(SDL_DestroyTexture);
DEFINE_SDL2_POINTER(SDL_DestroyWindow);
DEFINE_SDL2_POINTER(SDL_FreeSurface);
DEFINE_SDL2_POINTER(SDL_GetError);
DEFINE_SDL2_POINTER(SDL_GetTicks);
DEFINE_SDL2_POINTER(SDL_InitSubSystem);
DEFINE_SDL2_POINTER(SDL_LockAudio);
DEFINE_SDL2_POINTER(SDL_OpenAudio);
DEFINE_SDL2_POINTER(SDL_PauseAudio);
DEFINE_SDL2_POINTER(SDL_PollEvent);
DEFINE_SDL2_POINTER(SDL_Quit);
DEFINE_SDL2_POINTER(SDL_RenderCopy);
DEFINE_SDL2_POINTER(SDL_RenderPresent);
DEFINE_SDL2_POINTER(SDL_RenderSetIntegerScale);
DEFINE_SDL2_POINTER(SDL_RenderSetLogicalSize);
DEFINE_SDL2_POINTER(SDL_UnlockAudio);
DEFINE_SDL2_POINTER(SDL_UpdateTexture);
DEFINE_SDL2_POINTER(SDL_WasInit);
DEFINE_SDL2_POINTER(SDL_GetCurrentDisplayMode);
DEFINE_SDL2_POINTER(SDL_GetRenderDriverInfo);
DEFINE_SDL2_POINTER(SDL_GetCurrentVideoDriver);

/* SDL_Init */
int SDL_Init(uint32_t flags) {
  LINK_NAMESPACE_SDL2(SDL_Init);
  return SDL2::SDL_Init(flags);
}

/* SDL_AllocFormat */
SDL_PixelFormat* SDL_AllocFormat(uint32_t pixelFormat) {
  LINK_NAMESPACE_SDL2(SDL_AllocFormat);
  return SDL2::SDL_AllocFormat(pixelFormat);
}

/* SDL_CloseAudio */
void SDL_CloseAudio(void) {
  LINK_NAMESPACE_SDL2(SDL_CloseAudio);
  return SDL2::SDL_CloseAudio();
}

/* SDL_CreateRGBSurface */
SDL_Surface* SDL_CreateRGBSurface(uint32_t flags,
                                  int      width,
                                  int      height,
                                  int      depth,
                                  uint32_t Rmask,
                                  uint32_t Gmask,
                                  uint32_t Bmask,
                                  uint32_t Amask) {
  LINK_NAMESPACE_SDL2(SDL_CreateRGBSurface);
  return SDL2::SDL_CreateRGBSurface(flags, width, height, depth, Rmask, Gmask, Bmask, Amask);
}

/* SDL_CreateRenderer */
SDL_Renderer* SDL_CreateRenderer(SDL_Window* window,
                                 int         index,
                                 uint32_t    flags) {
  LINK_NAMESPACE_SDL2(SDL_CreateRenderer);
  return SDL2::SDL_CreateRenderer(window, index, flags);
}

/* SDL_CreateTexture */
SDL_Texture* SDL_CreateTexture(SDL_Renderer* renderer,
                               uint32_t      format,
                               int           access,
                               int           w,
                               int           h) {
  LINK_NAMESPACE_SDL2(SDL_CreateTexture);
  return SDL2::SDL_CreateTexture(renderer, format, access, w, h);
}

/* SDL_CreateWindow */
SDL_Window* SDL_CreateWindow(const char* title,
                             int         x,
                             int         y,
                             int         w,
                             int         h,
                             uint32_t    flags) {
  LINK_NAMESPACE_SDL2(SDL_CreateWindow);
  return SDL2::SDL_CreateWindow(title, x, y, w, h, flags);
}

/* SDL_Delay */
void SDL_Delay(uint32_t ms) {
  LINK_NAMESPACE_SDL2(SDL_Delay);
  return SDL2::SDL_Delay(ms);
}

/* SDL_DestroyRenderer */
void SDL_DestroyRenderer(SDL_Renderer* renderer) {
  LINK_NAMESPACE_SDL2(SDL_DestroyRenderer);
  return SDL2::SDL_DestroyRenderer(renderer);
}

/* SDL_DestroyTexture */
void SDL_DestroyTexture(SDL_Texture* texture) {
  LINK_NAMESPACE_SDL2(SDL_DestroyTexture);
  return SDL2::SDL_DestroyTexture(texture);
}

/* SDL_DestroyWindow */
void SDL_DestroyWindow(SDL_Window* window) {
  LINK_NAMESPACE_SDL2(SDL_DestroyWindow);
  return SDL2::SDL_DestroyWindow(window);
}

/* SDL_FreeSurface */
void SDL_FreeSurface(SDL_Surface* surface) {
  LINK_NAMESPACE_SDL2(SDL_FreeSurface);
  return SDL2::SDL_FreeSurface(surface);
}

/* SDL_GetError */
const char* SDL_GetError(void) {
  LINK_NAMESPACE_SDL2(SDL_GetError);
  return SDL2::SDL_GetError();
}

/* SDL_GetTicks */
uint32_t SDL_GetTicks(void) {
  LINK_NAMESPACE_SDL2(SDL_GetTicks);
  return SDL2::SDL_GetTicks();
}

/* SDL_InitSubSystem */
int SDL_InitSubSystem(uint32_t flags) {
  LINK_NAMESPACE_SDL2(SDL_InitSubSystem);
  return SDL2::SDL_InitSubSystem(flags);
}

/* SDL_LockAudio */
void SDL_LockAudio(void) {
  LINK_NAMESPACE_SDL2(SDL_LockAudio);
  return SDL2::SDL_LockAudio();
}

/* SDL_OpenAudio */
int SDL_OpenAudio(SDL_AudioSpec* desired,
                  SDL_AudioSpec* obtained) {
  LINK_NAMESPACE_SDL2(SDL_OpenAudio);
  return SDL2::SDL_OpenAudio(desired, obtained);
}

/* SDL_PauseAudio */
void SDL_PauseAudio(int pause_on) {
  LINK_NAMESPACE_SDL2(SDL_PauseAudio);
  return SDL2::SDL_PauseAudio(pause_on);
}

/* SDL_PollEvent */
int SDL_PollEvent(SDL_Event* event) {
  LINK_NAMESPACE_SDL2(SDL_PollEvent);
  return SDL2::SDL_PollEvent(event);
}

/* SDL_Quit */
void SDL_Quit(void) {
  LINK_NAMESPACE_SDL2(SDL_Quit);
  return SDL2::SDL_Quit();
}

/* SDL_RenderCopy */
int SDL_RenderCopy(SDL_Renderer*   renderer,
                   SDL_Texture*    texture,
                   const SDL_Rect* srcrect,
                   const SDL_Rect* dstrect) {
  LINK_NAMESPACE_SDL2(SDL_RenderCopy);
  return SDL2::SDL_RenderCopy(renderer, texture, srcrect, dstrect);
}

/* SDL_RenderPresent */
void SDL_RenderPresent(SDL_Renderer* renderer) {
  LINK_NAMESPACE_SDL2(SDL_RenderPresent);
  return SDL2::SDL_RenderPresent(renderer);
}

/* SDL_RenderSetIntegerScale */
int SDL_RenderSetIntegerScale(SDL_Renderer* renderer,
                              SDL_bool      enable) {
  LINK_NAMESPACE_SDL2(SDL_RenderSetIntegerScale);
  return SDL2::SDL_RenderSetIntegerScale(renderer, enable);
}

/* SDL_RenderSetLogicalSize */
int SDL_RenderSetLogicalSize(SDL_Renderer* renderer,
                             int           w,
                             int           h) {
  LINK_NAMESPACE_SDL2(SDL_RenderSetLogicalSize);
  return SDL2::SDL_RenderSetLogicalSize(renderer, w, h);
}

/* SDL_UnlockAudio */
void SDL_UnlockAudio(void) {
  LINK_NAMESPACE_SDL2(SDL_UnlockAudio);
  return SDL2::SDL_UnlockAudio();
}

/* SDL_UpdateTexture */
int SDL_UpdateTexture(SDL_Texture*    texture,
                      const SDL_Rect* rect,
                      const void*     pixels,
                      int             pitch) {
  LINK_NAMESPACE_SDL2(SDL_UpdateTexture);
  return SDL2::SDL_UpdateTexture(texture, rect, pixels, pitch);

}

/* SDL_WasInit */
uint32_t SDL_WasInit(uint32_t flags) {
  LINK_NAMESPACE_SDL2(SDL_WasInit);
  return SDL2::SDL_WasInit(flags);
}

/* SDL_GetCurrentDisplayMode */
int SDL_GetCurrentDisplayMode(int              displayIndex,
                              SDL_DisplayMode* mode) {
  LINK_NAMESPACE_SDL2(SDL_GetCurrentDisplayMode);
  return SDL2::SDL_GetCurrentDisplayMode(displayIndex, mode);
}

/* SDL_GetRenderDriverInfo */
int SDL_GetRenderDriverInfo(int               index,
                            SDL_RendererInfo* info) {
  LINK_NAMESPACE_SDL2(SDL_GetRenderDriverInfo);
  return SDL2::SDL_GetRenderDriverInfo(index, info);
}

/* SDL_GetCurrentVideoDriver */
const char* SDL_GetCurrentVideoDriver(void) {
  LINK_NAMESPACE_SDL2(SDL_GetCurrentVideoDriver);
  return SDL2::SDL_GetCurrentVideoDriver();
}

} // namespace ALE

#endif // defined(SDL_DYNLOAD)
