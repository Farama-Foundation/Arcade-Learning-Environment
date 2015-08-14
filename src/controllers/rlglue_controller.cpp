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
 *  rlglue_controller.cpp
 *
 *  The RLGlueController class implements an RL-Glue interface. It's based off
 *   the FIFOController, but the internals are different. Most of the code here
 *   is taken from the custom C++ environment found in the C/C++ RL-Glue codec.
 **************************************************************************** */

#include "rlglue_controller.hpp"

#ifdef __USE_RLGLUE
#include <stdio.h>
#include <stdlib.h> // getenv
#include <cassert>

#include "../environment/ale_ram.hpp"
#include <rlglue/utils/C/RLStruct_util.h>

#include "../common/Log.hpp"

RLGlueController::RLGlueController(OSystem* _osystem) :
  ALEController(_osystem) {
  m_max_num_frames = m_osystem->settings().getInt("max_num_frames");
  if (m_osystem->settings().getBool("restricted_action_set")) {
    available_actions = m_settings->getMinimalActionSet();
  } else {
    available_actions = m_settings->getAllActions();
  }
  m_send_rgb = m_osystem->settings().getBool("send_rgb");
}

RLGlueController::~RLGlueController() {
}

void RLGlueController::run() {
  // First perform handshaking
  initRLGlue();

  // Main loop
  rlGlueLoop();

  // Cleanly terminate RL-Glue
  endRLGlue();
}

bool RLGlueController::isDone() {
  // Die once we reach enough samples
  return ((m_max_num_frames > 0 && m_environment.getFrameNumber() >= m_max_num_frames));
}

void RLGlueController::initRLGlue() {
  ale::Logger::Info << "Initializing ALE RL-Glue ..." << std::endl;

  // Taken from setup_rlglue_network
  const char* host = kLocalHost;
  short port = kDefaultPort;

  const char* envptr = 0;

  envptr = getenv("RLGLUE_PORT");
  if (envptr != 0) {
    port = strtol(envptr, 0, 10);
    if (port == 0) {
      port = kDefaultPort;
    }
  }

  rlBufferCreate(&m_buffer, 4096);

  m_connection = rlWaitForConnection(host, port, kRetryTimeout);

  rlBufferClear(&m_buffer);
  rlSendBufferData(m_connection, &m_buffer, kEnvironmentConnection);
}

void RLGlueController::endRLGlue() {
  // Taken from teardown_rlglue_network
  rlClose(m_connection);
  rlBufferDestroy(&m_buffer);
}

void RLGlueController::rlGlueLoop() {
  int envState = 0;
  bool error = false;

  // Modified from runEnvironmentEventLoop
  while (!isDone() && !error && envState != kRLTerm) {
    rlBufferClear(&m_buffer);
    rlRecvBufferData(m_connection, &m_buffer, &envState);

    // Switch statement fills m_buffer with some data for RL-Glue
    switch(envState) {
      case kEnvInit:
        envInit();
        break;

      case kEnvStart:
        envStart();
        break;

      case kEnvStep:
        envStep();
        break;

      case kEnvCleanup:
        envCleanup();
        break;

      case kEnvMessage:
        envMessage();
        break;

      case kRLTerm:
        break;

      default:
        ale::Logger::Error << "Unknown RL-Glue command: " << envState << std::endl;
        error = true;
        break;
    };

    // Send back whatever we put in the buffer to the RL-Glue
    rlSendBufferData(m_connection, &m_buffer, envState);

    display();
  }
}

/** Initializes the environment; sends a 'task spec' */
void RLGlueController::envInit() {
  unsigned int offset = 0;
  unsigned int observation_dimensions;
  std::stringstream taskSpec;
  taskSpec << "VERSION RL-Glue-3.0 "
    "PROBLEMTYPE episodic "
    "DISCOUNTFACTOR 1 " // Goal is to maximize score... avoid unpleasant tradeoffs with 1 
    "OBSERVATIONS INTS (128 0 255)"; //RAM
  if (m_send_rgb) {
    taskSpec << "(100800 0 255) "; // Screen specified as an RGB triple per pixel
    observation_dimensions = 128 + 210 * 160 * 3;
  } else {
    taskSpec << "(33600 0 127) "; // Screen specified as one pallette index per pixel
    observation_dimensions = 128 + 210 * 160;
  }
  taskSpec << "ACTIONS INTS (0 " << available_actions.size() << ") "
    "REWARDS (UNSPEC UNSPEC) " // While rewards are technically bounded, this is safer 
    "EXTRA Name: Arcade Learning Environment ";
  // Allocate...?
  allocateRLStruct(&m_rlglue_action, 1, 0, 0);
  allocateRLStruct(&m_observation, observation_dimensions, 0, 0);
  // First write the task-spec length
  rlBufferClear(&m_buffer);
  unsigned int taskSpecLength = taskSpec.str().size();
  offset += rlBufferWrite(&m_buffer, offset, &taskSpecLength, 1, sizeof(int));
  // Then the string itself
  rlBufferWrite(&m_buffer, offset, taskSpec.str().c_str(), taskSpecLength, sizeof(char));
}

/** Sends the first observation out -- beginning an episode */
void RLGlueController::envStart() {
  // Reset the environment
  m_environment.reset();

  // Create the observation (we don't need reward/terminal here, but it's easier this way)
  reward_t reset_reward = 0;
  constructRewardObservationTerminal(reset_reward);

  // Copy into buffer
  rlBufferClear(&m_buffer);
  rlCopyADTToBuffer(&m_observation, &m_buffer, 0);
}

/** Reads in an action, returns the next observation-reward-terminal tuple.
    derived from onEnvStep(). */
void RLGlueController::envStep() {
  unsigned int offset = 0;

  offset = rlCopyBufferToADT(&m_buffer, offset, &m_rlglue_action);
  __RL_CHECK_STRUCT(&m_rlglue_action);

  unsigned int player_a_action_index = m_rlglue_action.intArray[0];

  // Filter for actions outside the expected range
  if (player_a_action_index >= available_actions.size()) {
    player_a_action_index = 0;
  }
  Action player_a_action = available_actions[player_a_action_index];
  Action player_b_action = (Action) PLAYER_B_NOOP;

  // Filter out non-regular actions ... let RL-Glue deal with those
  filterActions(player_a_action, player_b_action);

  // Pass these actions to ALE
  reward_t reward = applyActions(player_a_action, player_b_action);

  reward_observation_terminal_t ro = constructRewardObservationTerminal(reward);

  // Copy relevant data into the buffer
  rlBufferClear(&m_buffer);
  offset = 0;
  offset = rlBufferWrite(&m_buffer, offset, &ro.terminal, 1, sizeof(int));
  offset = rlBufferWrite(&m_buffer, offset, &ro.reward, 1, sizeof(double));
  offset = rlCopyADTToBuffer(ro.observation, &m_buffer, offset);
}

/** Performs some RL-Glue related cleanup. Adapted from oEnvCleanup(). */
void RLGlueController::envCleanup() {
  // Free data structures
  rlBufferClear(&m_buffer);
  clearRLStruct(&m_observation);
}

/** RL-Glue custom messages. Adapted from oEnvMessage(). */
void RLGlueController::envMessage() {
  unsigned int messageLength;
  unsigned int offset = 0;

  offset = rlBufferRead(&m_buffer, offset, &messageLength, 1, sizeof(int));
  // This could, of course, be stored somewhere for efficiency reasons
  if (messageLength > 0) {
    char * message = new char[messageLength+1];
    rlBufferRead(&m_buffer, offset, message, messageLength, sizeof(char));
    // Null terminate the string :(
    message[messageLength] = 0;

    ale::Logger::Error << "Message from RL-Glue: " << message << std::endl;

    delete[] message;
  }
}

void RLGlueController::filterActions(Action& player_a_action, Action& player_b_action) {
  if (player_a_action >= PLAYER_A_MAX)
    player_a_action = PLAYER_A_NOOP;
  if (player_b_action < PLAYER_B_NOOP || player_b_action >= PLAYER_B_MAX)
    player_b_action = PLAYER_B_NOOP;
}

reward_observation_terminal_t RLGlueController::constructRewardObservationTerminal(reward_t reward) {
  reward_observation_terminal_t ro;

  int index = 0;
  const ALERAM & ram = m_environment.getRAM();
  const ALEScreen & screen = m_environment.getScreen();

  // Copy RAM and screen into our big int-vector observation
  for (size_t i = 0; i < ram.size(); i++)
    m_observation.intArray[index++] = ram.get(i);

  size_t arraySize = screen.arraySize();

  if (m_send_rgb) {
    // Make sure we've allocated enough space for this
    assert (arraySize * 3 + ram.size() == m_observation.numInts);

    pixel_t *screenArray = screen.getArray();
    int red, green, blue;
    for (size_t i = 0; i < arraySize; i++) {
      m_osystem->colourPalette().getRGB(screenArray[i], red, green, blue);
      m_observation.intArray[index++] = red;
      m_observation.intArray[index++] = green;
      m_observation.intArray[index++] = blue;
    }
  } else {
    assert (arraySize + ram.size() == m_observation.numInts);
    for (size_t i = 0; i < arraySize; i++)
      m_observation.intArray[index++] = screen.getArray()[i];
  }

  ro.observation = &m_observation;

  // Fetch reward, terminal from the game settings
  ro.reward = reward;
  ro.terminal = m_settings->isTerminal();

  __RL_CHECK_STRUCT(ro.observation)

  return ro;
}

#else

RLGlueController::RLGlueController(OSystem* system):
  ALEController(system) {
}

void RLGlueController::run() {
  ale::Logger::Error << "RL-Glue interface unavailable. Please recompile with RL-Glue support." << 
    std::endl;

  // We should return and terminate gracefully, since we can.
}
#endif // __USE_RLGLUE
