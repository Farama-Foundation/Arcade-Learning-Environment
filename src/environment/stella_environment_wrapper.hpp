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

#include "../common/Constants.h"

class StellaEnvironment;

class StellaEnvironmentWrapper {
  // A wrapper for actions within the StellaEnvironment.
  // Allows us to call environment methods without requiring to #include
  // stella_environment.hpp.
  public:
    StellaEnvironmentWrapper(StellaEnvironment &environment);
    reward_t act(Action player_a_action, Action player_b_action);
    void softReset();
    void pressSelect(size_t num_steps = 1);
    
    StellaEnvironment &m_environment;
};

#endif // __STELLA_ENVIRONMENT_WRAPPER_HPP__

