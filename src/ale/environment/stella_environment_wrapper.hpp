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
 *  stella_environment_wrapper.hpp
 *
 * Method wrapper for StellaEnvironment.
 *
 **************************************************************************** */

#ifndef __STELLA_ENVIRONMENT_WRAPPER_HPP__
#define __STELLA_ENVIRONMENT_WRAPPER_HPP__

#include "ale/common/Constants.h"
#include "ale/emucore/Random.hxx"

namespace ale {

class StellaEnvironment;

class StellaEnvironmentWrapper {
  // A wrapper for actions within the StellaEnvironment.
  // Allows us to call environment methods without requiring to #include
  // stella_environment.hpp.
 public:
  StellaEnvironmentWrapper(StellaEnvironment& environment);
  reward_t act(Action player_a_action, Action player_b_action,
               float paddle_a_strength = 1.0, float paddle_b_strength = 1.0);
  void softReset();
  void pressSelect(size_t num_steps = 1);
  stella::Random& getEnvironmentRNG();

  StellaEnvironment& m_environment;
};

}  // namespace ale

#endif  // __STELLA_ENVIRONMENT_WRAPPER_HPP__
