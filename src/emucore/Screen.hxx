/******************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  Screen.hxx
 *
 *  Base class for displaying the screen of the 2600.
 **************************************************************************** */

#ifndef SCREEN_HXX
#define SCREEN_HXX

namespace ale {
namespace stella {

class OSystem;

class Screen
{
  public:
    Screen(OSystem* osystem) { myOSystem = osystem; }
    virtual ~Screen() { };

  public:
    virtual void render() { };

  protected:
    OSystem* myOSystem;
};

}  // namespace stella
}  // namespace ale

#endif
