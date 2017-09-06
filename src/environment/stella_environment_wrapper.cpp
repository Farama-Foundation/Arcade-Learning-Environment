#include "stella_environment.hpp"
#include "stella_environment_wrapper.hpp"

StellaEnvironmentWrapper::StellaEnvironmentWrapper(StellaEnvironment &environment) :
    m_environment(environment) {
}

reward_t StellaEnvironmentWrapper::act(Action player_a_action, Action player_b_action) {
    return m_environment.act(player_a_action, player_b_action);
}

void StellaEnvironmentWrapper::softReset() {
    m_environment.softReset();
}

void StellaEnvironmentWrapper::pressSelect(size_t num_steps) {
    m_environment.pressSelect(num_steps);
}
