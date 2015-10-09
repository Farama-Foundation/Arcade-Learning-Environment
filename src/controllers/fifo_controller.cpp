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
 *  fifo_controller.cpp
 *
 *  The FIFOController class implements an Agent/ALE interface via stdin/stdout
 *  or named pipes.
 **************************************************************************** */

#include "fifo_controller.hpp"

#include <stdio.h>
#include <cassert>
#include "../common/Log.hpp"

#define MAX_RUN_LENGTH (0xFF)

static const char hexval[] = { 
    '0', '1', '2', '3', '4', '5', '6', '7', 
    '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' 
};

/* appends a pixels value to the string buffer, returning the number of characters written */
inline void appendByte(char *buf, uInt8 v) {
    *buf = hexval[(v >> 4)];
    *(buf+1) = hexval[v & 0xF];
}

FIFOController::FIFOController(OSystem* _osystem, bool named_pipes) :
  ALEController(_osystem),
  m_named_pipes(named_pipes) {
  m_max_num_frames = m_osystem->settings().getInt("max_num_frames");
  m_run_length_encoding = m_osystem->settings().getBool("run_length_encoding");
}

FIFOController::~FIFOController() {
  if (m_fout != NULL) fclose(m_fout);
  if (m_fin != NULL) fclose(m_fin);
}

void FIFOController::run() {
  Action action_a, action_b;

  // First perform handshaking
  handshake();

  // Main loop
  while (!isDone()) { 
    // Send data over to agent
    sendData();
    // Read agent's response & process it
    readAction(action_a, action_b);

    // Emulate Atari forward
    latest_reward = applyActions(action_a, action_b);

    // Update display if needed
    display();
  }

  // Send a termination signal to the agent, if they're still around
  if (!feof(m_fout))
    fprintf (m_fout, "DIE\n");
}

bool FIFOController::isDone() {
  // Die once we reach enough samples
  return ((m_max_num_frames > 0 && m_environment.getFrameNumber() >= m_max_num_frames) ||
    feof(m_fin) || feof(m_fout) || ferror(m_fout));
}

void FIFOController::handshake() {
  // If using named pipes, open said files
  if (m_named_pipes) {
    openNamedPipes();
  } else { // Otherwise read from stdin and output to stdout
      m_fout = stdout;
      m_fin = stdin;
      assert(m_fin != NULL && m_fout != NULL);
  }

  // send the width and height of the screen through the pipe
  char out_buffer [1024];
  
  snprintf (out_buffer, sizeof(out_buffer), "%d-%d\n", 
    (int)m_environment.getScreen().width(),
    (int)m_environment.getScreen().height());
  
  fputs(out_buffer, m_fout);
  fflush (m_fout);

  // Read in agent's response
  char in_buffer [1024];
  fgets (in_buffer, sizeof(in_buffer), m_fin);

  // Parse response: send_screen, send_ram, <obsolete>, send_RL
  char * token = strtok (in_buffer,",\n");
  m_send_screen = atoi(token);
  token = strtok (NULL,",\n");
  m_send_ram = atoi(token);
  token = strtok (NULL,",\n");
  // Used to be frame skip; now obsolete
  token = strtok(NULL, ",\n");
  m_send_rl = atoi(token);
}

void FIFOController::openNamedPipes() {
  m_fout = fopen("ale_fifo_out", "w");
  if (m_fout == NULL) {
    ale::Logger::Error << "Missing output pipe: ale_fifo_out" << std::endl;
    exit(1);
  }

  m_fin = fopen("ale_fifo_in", "r");

  if (m_fin == NULL) {
    ale::Logger::Error << "Missing output pipe: ale_fifo_out" << std::endl;
    exit(1);
  }
}

void FIFOController::sendData() {
  if (m_send_ram) sendRAM();
  if (m_send_screen) sendScreen();
  if (m_send_rl) sendRL();
  // Send the terminating newline
  fputc('\n', m_fout);
  fflush(m_fout);
}

void FIFOController::sendScreen() {
  // Obtain the screen from the environment 
  const ALEScreen& screen = m_environment.getScreen();

  char buffer[204800];
  int sn;

  // Encode the screen into a char buffer
  if (m_run_length_encoding)
    sn = stringScreenRLE(screen, buffer);
  else
    sn = stringScreenFull(screen, buffer);

  // Append terminating stuff, send
  buffer[sn] = ':';
  buffer[sn+1] = 0;

  fputs(buffer, m_fout);
}

int FIFOController::stringScreenRLE(const ALEScreen& screen, char* buffer) {
  int currentColor = -1;
  int runLength = 0;

  int sn = 0;

  // Process pixels in array-order
  for (size_t i = 0; i < screen.arraySize(); i++) {
    pixel_t col = screen.getArray()[i];

    // Lengthen this run
    if (col == currentColor && runLength < MAX_RUN_LENGTH)
      runLength++;
    else {
      if (currentColor != -1) {
        // Output it
        appendByte(buffer + sn, currentColor);
        appendByte(buffer + sn + 2, runLength);
        sn += 4;
      }

      // Switch to the new color
      currentColor = col;
      runLength = 1;
    }
  }

  appendByte(buffer + sn, currentColor);
  appendByte(buffer + sn + 2, runLength);
  sn += 4;

  return sn;
}

int FIFOController::stringScreenFull(const ALEScreen& screen, char* buffer) {
  int sn = 0;

  // Iterate through pixels, put in a string
  for (size_t i = 0; i < screen.arraySize(); i++) {
    pixel_t col = screen.getArray()[i];

    appendByte(buffer + sn, col);
    sn += 2;
  }

  return sn;
}

void FIFOController::sendRAM() {
  const ALERAM& ram = m_environment.getRAM();

  char buffer[204800];
  int sn = 0;

  // Convert the RAM bytes into a string
  for (size_t i = 0; i < ram.size(); i++) {
    byte_t b = ram.get(i);
    appendByte(buffer + sn, b);
    sn += 2;
  }

  // Output RAM
  buffer[sn] = ':';
  buffer[sn+1] = 0;
  fputs(buffer, m_fout);
}

void FIFOController::sendRL() {
  int r = (int) latest_reward;
  bool is_terminal = m_environment.isTerminal();

  fprintf(m_fout, "%d,%d:", is_terminal, r);
}

void FIFOController::readAction(Action& action_a, Action& action_b) {
  // Read the new action from the pipe, as a comma-separated pair
  char in_buffer[2048];
  fgets (in_buffer, sizeof(in_buffer), m_fin);
 
  char * token = strtok (in_buffer,",\n");
  action_a = (Action)atoi(token);

  token = strtok (NULL,",\n");
  action_b = (Action)atoi(token);
}

