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
 *  phosphor_blend.hpp
 *
 *  Methods for performing colour averaging over the screen.
 *
 **************************************************************************** */

#ifndef __PHOSPHOR_BLEND_HPP__
#define __PHOSPHOR_BLEND_HPP__

#include "ale/emucore/OSystem.hxx"
#include "ale/environment/ale_screen.hpp"

namespace ale {

class PhosphorBlend {
 public:
  PhosphorBlend(stella::OSystem*);

  void process(ALEScreen& screen);

 private:
  void makeAveragePalette();
  uint8_t getPhosphor(uint8_t v1, uint8_t v2);
  uint32_t makeRGB(uint8_t r, uint8_t g, uint8_t b);
  /** Converts a RGB value to an 8-bit format */
  uint8_t rgbToNTSC(uint32_t rgb);

 private:
  stella::OSystem* m_osystem;

  uint8_t m_rgb_ntsc[64][64][64];

  uint32_t m_avg_palette[256][256];
  uint8_t m_phosphor_blend_ratio;
};

}  // namespace ale

#endif  // __PHOSPHOR_BLEND_HPP__
