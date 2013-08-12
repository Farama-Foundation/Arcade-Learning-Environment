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

// How many observation dimensions
#define NUM_OBSERVATION_DIMENSIONS (128 + 210*160)

RLGlueController::RLGlueController(OSystem* _osystem) :
  ALEController(_osystem) {
  m_max_num_frames = m_osystem->settings().getInt("max_num_frames");
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
  std::cerr << "Initializing ALE RL-Glue ..." << std::endl;

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
        std::cerr << "Unknown RL-Glue command: " << envState << std::endl;
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
  unsigned int taskSpecLength = 0;
  unsigned int offset = 0;
 
  // Possibly this should be one big snprintf.
  std::string taskSpec = std::string("") +
    "VERSION RL-Glue-3.0 "+
    "PROBLEMTYPE episodic "+
    "DISCOUNTFACTOR 1 "+ // Goal is to maximize score... avoid unpleasant tradeoffs with 1 
    "OBSERVATIONS INTS (128 0 255)(33600 0 127) "+ // RAM, then screen
    //"ACTIONS INTS (0 17) "+ // Inactive PlayerB 
    "ACTIONS INTS (0 17)(18 35) "+ // Two actions: player A and player B
    "REWARDS (UNSPEC UNSPEC) "+ // While rewards are technically bounded, this is safer 
    "EXTRA Name: Arcade Learning Environment ";

  taskSpecLength = taskSpec.length();
 
  // Allocate...? 
  allocateRLStruct(&m_rlglue_action, 2, 0, 0);
  allocateRLStruct(&m_observation, NUM_OBSERVATION_DIMENSIONS, 0, 0);

  // First write the task-spec length
  rlBufferClear(&m_buffer);
  offset += rlBufferWrite(&m_buffer, offset, &taskSpecLength, 1, sizeof(int));
  // Then the string itself
  rlBufferWrite(&m_buffer, offset, taskSpec.c_str(), taskSpec.length(), sizeof(char));
}

/** Sends the first observation out -- beginning an episode */
void RLGlueController::envStart() {
  // Reset the environment
  m_environment.reset();

  // Create the observation (we don't need reward/terminal here, but it's easier this way)
  reward_t reset_reward = 0;
  reward_observation_terminal_t ro = constructRewardObservationTerminal(reset_reward);

  // Copy into buffer
  rlBufferClear(&m_buffer);
  rlCopyADTToBuffer(&m_observation, &m_buffer, 0);
}

/** Reads in an action, returns the next observation-reward-terminal tuple.
  *  derived from onEnvStep(). */ 
void RLGlueController::envStep() {
  unsigned int offset = 0;
 
  offset = rlCopyBufferToADT(&m_buffer, offset, &m_rlglue_action);
  __RL_CHECK_STRUCT(&m_rlglue_action);

  // We expect here an integer-valued action
  Action player_a_action = (Action)m_rlglue_action.intArray[0];
  Action player_b_action = (Action)m_rlglue_action.intArray[1]; 

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

    std::cerr << "Message from RL-Glue: " << message << std::endl;

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
  for (size_t i = 0; i < screen.arraySize(); i++)
    m_observation.intArray[index++] = screen.getArray()[i];

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
  std::cerr << "RL-Glue interface unavailable. Please recompile with RL-Glue support." << 
    std::endl;

  // We should return and terminate gracefully, since we can.
}
#endif // __USE_RLGLUE
