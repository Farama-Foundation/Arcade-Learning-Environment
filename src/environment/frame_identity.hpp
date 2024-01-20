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
 *  frame_identity.hpp
 *
 *  Frame maxing processor.
 *
 **************************************************************************** */

#ifndef __FRAME_IDENTITY_HPP__
#define __FRAME_IDENTITY_HPP__

#include <algorithm>
#include <cstdint>

#include "emucore/MediaSrc.hxx"
#include "common/ColourPalette.hpp"
#include "environment/frame_processor.hpp"

namespace ale {

class FrameIdentity : public FrameProcessor {
 public:
  FrameIdentity(ColourPalette& palette) : FrameProcessor(palette) {}

  inline void processGrayscale(
    stella::MediaSource& media,
    uint8_t* out
  ) {
    uint8_t* currentFrameBuffer = media.currentFrameBuffer();
    size_t frameSize = media.width() * media.height();
    m_palette.applyPaletteGrayscale(out, currentFrameBuffer, frameSize);
  }

  inline void processRGB(
    stella::MediaSource& media,
    uint8_t* out
  ) {
    uint8_t* currentFrameBuffer = media.currentFrameBuffer();
    size_t frameSize = media.width() * media.height();
    m_palette.applyPaletteRGB(out, currentFrameBuffer, frameSize);
  }
};

}  // namespace ale

#endif  // __FRAME_IDENTITY__
