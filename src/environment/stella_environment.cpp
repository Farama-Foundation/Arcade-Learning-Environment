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
 *  stella_environment.cpp
 *
 *  A class that wraps around the Stella core to provide users with a typical
 *  reinforcement learning environment interface.
 *  
 **************************************************************************** */

#include "stella_environment.hpp"
#include "../emucore/m6502/src/System.hxx"

StellaEnvironment::StellaEnvironment(OSystem* osystem, RomSettings* settings):
  m_osystem(osystem),
  m_settings(settings),
  m_phosphor_blend(osystem),
  m_frame_number(0),
  m_episode_frame_number(0),
  m_screen(m_osystem->console().mediaSource().height(),
        m_osystem->console().mediaSource().width()) {

  // Determine whether this is a paddle-based game
  if (m_osystem->console().properties().get(Controller_Left) == "PADDLES" ||
      m_osystem->console().properties().get(Controller_Right) == "PADDLES") {
    m_use_paddles = true;
    m_state.resetPaddles(m_osystem->event());
  } else {
    m_use_paddles = false;
  }

  m_cartridge_md5 = m_osystem->console().properties().get(Cartridge_MD5);
  
  m_num_reset_steps = atoi(m_osystem->settings().getString("system_reset_steps").c_str());
  m_use_starting_actions = m_osystem->settings().getBool("use_starting_actions");
  
  m_max_num_frames_per_episode = m_osystem->settings().getInt("max_num_frames_per_episode");
  m_colour_averaging = !m_osystem->settings().getBool("disable_color_averaging");

  m_backward_compatible_save = m_osystem->settings().getBool("backward_compatible_save");
  m_stochastic_start = m_osystem->settings().getBool("use_environment_distribution");
  
  m_frame_skip = m_osystem->settings().getInt("frame_skip");
  if (m_frame_skip < 1) {
    fprintf (stderr, "Warning: frame skip set to < 1. Setting to 1.\n");
    m_frame_skip = 1;
  }
}

/** Resets the system to its start state. */
void StellaEnvironment::reset() {
  // RNG for generating environments
  Random randGen;

  m_episode_frame_number = 0;

  // Reset the paddles
  m_state.resetPaddles(m_osystem->event());

  // Reset the emulator
  m_osystem->console().system().reset();

  // NOOP for 60 steps in the deterministic environment setting, or some random amount otherwise 
  int noopSteps;
  if (m_stochastic_start)
    noopSteps = 60 + rand() % NUM_RANDOM_ENVIRONMENTS;
  else
    noopSteps = 60;

  emulate(PLAYER_A_NOOP, PLAYER_B_NOOP, noopSteps);
  // reset for n steps
  emulate(RESET, PLAYER_B_NOOP, m_num_reset_steps);

  // reset the rom (after emulating, in case the NOOPs led to reward)
  m_settings->reset();
  
  // Apply necessary actions specified by the rom itself
  if (m_use_starting_actions) {
    ActionVect startingActions = m_settings->getStartingActions();
    for (size_t i = 0; i < startingActions.size(); i++)
      emulate(startingActions[i], PLAYER_B_NOOP);
  }
}

/** Save/restore the environment state. */
void StellaEnvironment::save() {
  // Store the current state into a new object
  ALEState new_state = cloneState();

  if (m_backward_compatible_save) { // 0.2, 0.3: overwrite on save
    while (!m_saved_states.empty())
      m_saved_states.pop();
    m_saved_states.push(new_state);
  }
  else { // 0.4 and above: put it on the stack
    m_saved_states.push(new_state);
  }
}

void StellaEnvironment::load() {
  // Get the state on top of the stack
  ALEState& target_state = m_saved_states.top(); 
 
  // Deserialize it into 'm_state'
  restoreState(target_state);

  if (m_backward_compatible_save) { // 0.2, 0.3: persistent save 
  }
  else { // 0.4 and above: take it off the stack
    m_saved_states.pop();
  }
}

/** Returns a copy of the current emulator state. */
ALEState StellaEnvironment::cloneState() {
  return m_state.save(m_osystem, m_settings, m_cartridge_md5);
}

/** Restores a previously saved copy of the state. */
void StellaEnvironment::restoreState(const ALEState& target_state) {
  m_state.load(m_osystem, m_settings, m_cartridge_md5, target_state);
}


void StellaEnvironment::noopIllegalActions(Action & player_a_action, Action & player_b_action) {
  if (player_a_action < (Action)PLAYER_B_NOOP && 
        !m_settings->isLegal(player_a_action)) {
    player_a_action = (Action)PLAYER_A_NOOP;
  }
  // Also drop RESET, which doesn't play nice with our clean notions of RL environments
  else if (player_a_action == RESET) 
    player_a_action = (Action)PLAYER_A_NOOP;

  if (player_b_action < (Action)RESET && 
        !m_settings->isLegal((Action)((int)player_b_action - PLAYER_B_NOOP))) {
    player_b_action = (Action)PLAYER_B_NOOP;
  }
  else if (player_b_action == RESET) 
    player_b_action = (Action)PLAYER_B_NOOP;
}

reward_t StellaEnvironment::act(Action player_a_action, Action player_b_action) {
  // Total reward received as we repeat the action
  reward_t sum_rewards = 0;

  // Apply the same action for a given number of times... note that act() will refuse to emulate 
  //  past the terminal state
  for (size_t i = 0; i < m_frame_skip; i++)
    sum_rewards += oneStepAct(player_a_action, player_b_action);

  return sum_rewards;
}

/** Applies the given actions (e.g. updating paddle positions when the paddle is used)
  *  and performs one simulation step in Stella. */
reward_t StellaEnvironment::oneStepAct(Action player_a_action, Action player_b_action) {
  // Once in a terminal state, refuse to go any further (special actions must be handled
  //  outside of this environment; in particular reset() should be called rather than passing
  //  RESET or SYSTEM_RESET.
  if (isTerminal())
    return 0;

  // Convert illegal actions into NOOPs; actions such as reset are always legal
  noopIllegalActions(player_a_action, player_b_action);
  
  // Emulate in the emulator
  emulate(player_a_action, player_b_action);
  m_state.incrementFrame();

  // Increment the number of frames seen so far
  m_frame_number++;
  m_episode_frame_number++;

  return m_settings->getReward();
}

bool StellaEnvironment::isTerminal() {
  return (m_settings->isTerminal() || 
    (m_max_num_frames_per_episode > 0 && m_episode_frame_number >= m_max_num_frames_per_episode));
}

void StellaEnvironment::emulate(Action player_a_action, Action player_b_action, size_t num_steps) {
  Event* event = m_osystem->event();
  
  // Handle paddles separately: we have to manually update the paddle positions at each step
  if (m_use_paddles) {
    // Run emulator forward for 'num_steps'
    for (size_t t = 0; t < num_steps; t++) {
      // Update paddle position at every step
      m_state.applyActionPaddles(event, player_a_action, player_b_action);

      m_osystem->console().mediaSource().update();
      m_settings->step(m_osystem->console().system());
    }
  }
  else {
    // In joystick mode we only need to set the action events once
    m_state.setActionJoysticks(event, player_a_action, player_b_action);

    for (size_t t = 0; t < num_steps; t++) {
      m_osystem->console().mediaSource().update();
      m_settings->step(m_osystem->console().system());
    }
  }

  // Parse screen and RAM into their respective data structures
  processScreen();
  processRAM();
}

/** Accessor methods for the environment state. */
void StellaEnvironment::setState(const ALEState& state) {
  m_state = state;
}

const ALEState& StellaEnvironment::getState() const {
  return m_state;
}

void StellaEnvironment::processScreen() {
  if (!m_colour_averaging) {
    // Copy screen over and we're done! 
    memcpy(m_screen.getArray(), 
      m_osystem->console().mediaSource().currentFrameBuffer(), m_screen.arraySize());
  }
  else {
    // Perform phosphor averaging; the blender stores its result in the given screen
    m_phosphor_blend.process(m_screen);
  }
}

void StellaEnvironment::processRAM() {
  // Copy RAM over
  for (size_t i = 0; i < m_ram.size(); i++)
    *m_ram.byte(i) = m_osystem->console().system().peek(i + 0x80); 
}

