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

#include <sstream>
#include <cstring>

#include "emucore/System.hxx"

namespace ale {

StellaEnvironment::StellaEnvironment(OSystem* osystem, RomSettings* settings)
    : m_osystem(osystem),
      m_settings(settings),
      m_phosphor_blend(osystem),
      m_screen(m_osystem->console().mediaSource().height(),
               m_osystem->console().mediaSource().width()),
      m_actions(4, PLAYER_A_NOOP) {
  // Determine whether this is a paddle-based game
  if (m_osystem->console().properties().get(Controller_Left) == "PADDLES" ||
      m_osystem->console().properties().get(Controller_Right) == "PADDLES") {
    m_use_paddles = true;
    int paddle_min_val = m_osystem->settings().getInt("paddle_min");
    int paddle_max_val = m_osystem->settings().getInt("paddle_max");
    m_state.setPaddleLimits(paddle_min_val != -1 ? paddle_min_val : PADDLE_MIN,
                            paddle_max_val != -1 ? paddle_max_val : PADDLE_MAX);
    m_state.resetPaddles(m_osystem->event());
  } else {
    m_use_paddles = false;
  }
  m_num_reset_steps = 4;
  m_cartridge_md5 = m_osystem->console().properties().get(Cartridge_MD5);

  // Set current mode to the ROM's default mode
  setMode(settings->getDefaultMode());

  m_max_num_frames_per_episode =
      m_osystem->settings().getInt("max_num_frames_per_episode");
  m_colour_averaging = m_osystem->settings().getBool("color_averaging");

  m_repeat_action_probability =
      m_osystem->settings().getFloat("repeat_action_probability");

  m_frame_skip = m_osystem->settings().getInt("frame_skip");
  if (m_frame_skip < 1) {
    Logger::Warning << "Warning: frame skip set to < 1. Setting to 1.\n";
    m_frame_skip = 1;
  }

  // If so desired, we record all emulated frames to a given directory
  std::string recordDir = m_osystem->settings().getString("record_screen_dir");
  if (!recordDir.empty()) {
    Logger::Info << "Recording screens to directory: " << recordDir << "\n";

    // Create the screen exporter
    m_screen_exporter.reset(
        new ScreenExporter(m_osystem->colourPalette(), recordDir));
  }
}

/** Resets the system to its start state. */
void StellaEnvironment::reset() {
  m_state.resetEpisodeFrameNumber();
  // Reset the paddles
  m_state.resetPaddles(m_osystem->event());

  // Reset the emulator
  m_osystem->console().system().reset();

  // NOOP for 60 steps in the deterministic environment setting, or some random amount otherwise
  int noopSteps;
  noopSteps = 60;

  emulate(PLAYER_A_NOOP, PLAYER_B_NOOP, noopSteps);
  // Reset the emulator
  softReset();

  // reset the rom (after emulating, in case the NOOPs led to reward)
  m_settings->reset();

  // Apply mode that was previously defined, then soft reset with this mode
  m_settings->setMode(m_state.getCurrentMode(), m_osystem->console().system(),
                      getWrapper());
  softReset();

  // Apply necessary actions specified by the rom itself
  ActionVect startingActions = m_settings->getStartingActions();
  for (size_t i = 0; i < startingActions.size(); i++) {
    emulate({startingActions[i], startingActions[i]});
  }
}

/** Save/restore the environment state. */
void StellaEnvironment::save() {
  // Store the current state into a new object
  ALEState new_state = cloneState();
  m_saved_states.push(new_state);
}

void StellaEnvironment::load() {
  // Get the state on top of the stack
  ALEState& target_state = m_saved_states.top();

  // Deserialize it into 'm_state'
  restoreState(target_state);
  m_saved_states.pop();
}

ALEState StellaEnvironment::cloneState() {
  return m_state.save(m_osystem, m_settings, m_cartridge_md5, false);
}

void StellaEnvironment::restoreState(const ALEState& target_state) {
  m_state.load(m_osystem, m_settings, m_cartridge_md5, target_state, false);
}

ALEState StellaEnvironment::cloneSystemState() {
  return m_state.save(m_osystem, m_settings, m_cartridge_md5, true);
}

void StellaEnvironment::restoreSystemState(const ALEState& target_state) {
  m_state.load(m_osystem, m_settings, m_cartridge_md5, target_state, true);
}

void StellaEnvironment::noopIllegalAction(Action& action) {
  if ((!m_settings->isLegal(action) && action < (Action)PLAYER_B_NOOP) || action == RESET) {
      action = PLAYER_A_NOOP;
  }
}

reward_t StellaEnvironment::act(Action player_a_action,
                                Action player_b_action) {
  return act(std::vector<Action>{player_a_action,(Action)(player_b_action - PLAYER_B_NOOP)}).at(0);
}

std::vector<reward_t> StellaEnvironment::act(std::vector<Action> actions) {
  // Total reward received as we repeat the action
  std::vector<reward_t> sum_rewards(actions.size(),0);

  Random& rng = m_osystem->rng();

  // Apply the same action for a given number of times... note that act() will refuse to emulate
  //  past the terminal state
  for (size_t j = 0; j < m_frame_skip; j++) {
    // Stochastically drop actions, according to mm_repeat_action_probability
    for (size_t i = 0; i < 4; i++) {
      if (i < actions.size()) {
        if (rng.nextDouble() >= m_repeat_action_probability)
          m_actions[i] = actions[i];
      }
      else {
        m_actions[i] = PLAYER_A_NOOP;
      }
    }
    oneStepAct(m_actions, sum_rewards);
  }

  return sum_rewards;
}

/** This functions emulates a push on the reset button of the console */
void StellaEnvironment::softReset() {
  emulate(RESET, PLAYER_B_NOOP, m_num_reset_steps);

  // Reset previous actions to NOOP for correct action repeating
  for (Action & a : m_actions) {
    a = PLAYER_A_NOOP;
  }
}

/** Applies the given actions (e.g. updating paddle positions when the paddle is used)
 *  and performs one simulation step in Stella. */
void StellaEnvironment::oneStepAct(std::vector<Action> actions,std::vector<reward_t> & rewards) {
  // Once in a terminal state, refuse to go any further (special actions must be handled
  //  outside of this environment; in particular reset() should be called rather than passing
  //  RESET or SYSTEM_RESET.
  if (isTerminal())
    return;

  // If so desired, request one frame's worth of sound (this does nothing if recording
  // is not enabled)
  m_osystem->sound().recordNextFrame();

  // Similarly record screen as needed
  if (m_screen_exporter.get() != NULL)
    m_screen_exporter->saveNext(m_screen);

  // Convert illegal actions into NOOPs; actions such as reset are always legal
  for(Action & a : actions){
    noopIllegalAction(a);
  }

  // Emulate in the emulator
  emulate(actions);
  // Increment the number of frames seen so far
  m_state.incrementFrame();

  rewards.at(0) += m_settings->getReward();
  if(rewards.size() > 1){
    rewards.at(1) += m_settings->getRewardP2();
  }
  if(rewards.size() > 2){
    rewards.at(2) += m_settings->getRewardP3();
  }
  if(rewards.size() > 3){
    rewards.at(3) += m_settings->getRewardP4();
  }
}

bool StellaEnvironment::isTerminal() const {
  return (m_settings->isTerminal() ||
          (m_max_num_frames_per_episode > 0 &&
           m_state.getEpisodeFrameNumber() >= m_max_num_frames_per_episode));
}

void StellaEnvironment::pressSelect(size_t num_steps) {
  m_state.pressSelect(m_osystem->event());
  for (size_t t = 0; t < num_steps; t++) {
    m_osystem->console().mediaSource().update();
  }
  processScreen();
  processRAM();
  emulate(PLAYER_A_NOOP, PLAYER_B_NOOP);
  m_state.incrementFrame();
}

void StellaEnvironment::setDifficulty(difficulty_t value) {
  m_state.setDifficulty(value);
}

// helper function for setMode
bool in_modes(const ModeVect & modes, game_mode_t m){
  return std::find(modes.begin(), modes.end(), m) != modes.end();
}

void StellaEnvironment::setMode(game_mode_t value) {
  int num_players;
  if (in_modes(m_settings->getAvailableModes(), value)) {
    num_players = 1;
  }
  else if (in_modes(m_settings->get2PlayerModes(), value)) {
    num_players = 2;
  }
  else if(in_modes(m_settings->get4PlayerModes(), value)){
    num_players = 4;
  }
  else {
    throw std::runtime_error("Invalid game mode requested");
  }
  m_state.setNumActivePlayers(num_players);
  m_state.setCurrentMode(value);
}

void StellaEnvironment::emulate(Action player_a_action, Action player_b_action,
                                size_t num_steps) {
  emulate({player_a_action,(Action)(player_b_action - PLAYER_B_NOOP)},num_steps);
}
void StellaEnvironment::emulate(std::vector<Action> actions,
                                size_t num_steps) {
  Event* event = m_osystem->event();
  for(Action a : actions){
    assert ((a < PLAYER_B_NOOP || a >= RESET) && "Actions in multiplayer cannot use the PLAYER_B actions. Rather, action lists should indicate the player by the position in the input vector");
  }

  // Handle paddles separately: we have to manually update the paddle positions at each step
  if (m_use_paddles) {
    // Run emulator forward for 'num_steps'
    for (size_t t = 0; t < num_steps; t++) {
      // Update paddle position at every step
      m_state.resetKeys(event);
      for (size_t p = 0; p < actions.size(); p++) {
        m_state.applyActionPaddle(event, actions[p], p);
      }

      m_osystem->console().mediaSource().update();
      m_settings->step(m_osystem->console().system());
    }
  } else {
    // In joystick mode we only need to set the action events once
    Action player_b_action = actions.size() >= 2 ? (Action)(actions[1] + PLAYER_B_NOOP) : PLAYER_B_NOOP;
    Action player_a_action = actions.at(0);
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
void StellaEnvironment::setState(const ALEState& state) { m_state = state; }

const ALEState& StellaEnvironment::getState() const { return m_state; }

std::unique_ptr<StellaEnvironmentWrapper> StellaEnvironment::getWrapper() {
  return std::unique_ptr<StellaEnvironmentWrapper>(
      new StellaEnvironmentWrapper(*this));
}

void StellaEnvironment::processScreen() {
  if (m_colour_averaging) {
    // Perform phosphor averaging; the blender stores its result in the given screen
    m_phosphor_blend.process(m_screen);
  } else {
    // Copy screen over and we're done!
    std::memcpy(m_screen.getArray(),
           m_osystem->console().mediaSource().currentFrameBuffer(),
           m_screen.arraySize());
  }
}

void StellaEnvironment::processRAM() {
  // Copy RAM over
  for (size_t i = 0; i < m_ram.size(); i++)
    *m_ram.byte(i) = m_osystem->console().system().peek(i + 0x80);
}

}  // namespace ale
