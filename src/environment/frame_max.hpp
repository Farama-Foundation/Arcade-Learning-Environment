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
 *  frame_max.hpp
 *
 *  Frame maxing processor.
 *
 **************************************************************************** */

#ifndef __FRAME_MAX_HPP__
#define __FRAME_MAX_HPP__

#include <algorithm>
#include <cstdint>

#include "emucore/MediaSrc.hxx"
#include "common/ColourPalette.hpp"
#include "environment/frame_processor.hpp"

namespace ale {

class FrameMax : public FrameProcessor {
 public:
  FrameMax(ColourPalette& palette) : FrameProcessor(palette) {}

  inline void processGrayscale(
    stella::MediaSource& media,
    uint8_t* out
  ) {
    const uint8_t* currentFrameBuffer = media.currentFrameBuffer();
    const uint8_t* previousFrameBuffer = media.previousFrameBuffer();
    const size_t frameSize = media.width() * media.height();

    for (size_t i = 0; i < frameSize; ++i) {
      uint8_t currentPixel = m_palette.getGrayscale(currentFrameBuffer[i]);
      uint8_t previousPixel = m_palette.getGrayscale(previousFrameBuffer[i]);
      out[i] = std::max(currentPixel, previousPixel);
    }
  }

  inline void processRGB(
    stella::MediaSource& media,
    uint8_t* out
  ) {
    const uint8_t* currentFrameBuffer = media.currentFrameBuffer();
    const uint8_t* previousFrameBuffer = media.previousFrameBuffer();
    const size_t frameSize = media.width() * media.height();

    for (size_t i = 0; i < frameSize; ++i) {
      uint32_t currentPixel = m_palette.getRGB(currentFrameBuffer[i]);
      uint32_t previousPixel = m_palette.getRGB(previousFrameBuffer[i]);

      *out = (uint8_t)std::max(currentPixel >> 16, previousPixel >> 16); out++;
      *out = (uint8_t)std::max(currentPixel >> 8, previousPixel >> 8); out++;
      *out = (uint8_t)std::max(currentPixel >> 0, previousPixel >> 0); out++;
    }
  }
};

}  // namespace ale

#endif  // __FRAME_MAX__
