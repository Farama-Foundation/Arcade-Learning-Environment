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
 *  frame_processor.hpp
 *
 *  Base class to postprocess a frame.
 *
 **************************************************************************** */

#ifndef __FRAME_PROCESSOR_HPP__
#define __FRAME_PROCESSOR_HPP__

#include <cstdint>

#include "emucore/MediaSrc.hxx"
#include "common/ColourPalette.hpp"

namespace ale {

class FrameProcessor {
 public:
  FrameProcessor(ColourPalette& palette) : m_palette(palette) {}
  virtual ~FrameProcessor() {}
  virtual void processGrayscale(
    stella::MediaSource& media,
    uint8_t* out
  ) = 0;
  virtual void processRGB(
    stella::MediaSource& media,
    uint8_t* out
  ) = 0;

 protected:
  ColourPalette& m_palette;
};

}  // namespace ale

#endif  // __FRAME_PROCESSOR__
