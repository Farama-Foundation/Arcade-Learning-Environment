#include "ale/environment/stella_environment_wrapper.hpp"

#include "ale/environment/stella_environment.hpp"

namespace ale {
using namespace stella;   // Random

StellaEnvironmentWrapper::StellaEnvironmentWrapper(
    StellaEnvironment& environment)
    : m_environment(environment) {}

reward_t StellaEnvironmentWrapper::act(Action player_a_action, Action player_b_action,
                                       float paddle_a_strength, float paddle_b_strength) {
  // Menu input from setMode: one frame, with no sticky actions, so the start
  // state does not depend on repeat_action_probability or frame_skip.
  return m_environment.act(player_a_action, player_b_action,
                           paddle_a_strength, paddle_b_strength, 0.0f, 1);
}

void StellaEnvironmentWrapper::softReset() { m_environment.softReset(); }

void StellaEnvironmentWrapper::pressSelect(size_t num_steps) {
  m_environment.pressSelect(num_steps);
}

Random& StellaEnvironmentWrapper::getEnvironmentRNG() {
  return m_environment.getEnvironmentRNG();
}

}  // namespace ale
