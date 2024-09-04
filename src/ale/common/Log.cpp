#include "ale/common/Log.hpp"

#include <iostream>

namespace ale {

Logger::mode Logger::current_mode = Info;

void Logger::setMode(Logger::mode m) { current_mode = m; }

Logger::mode operator<<(Logger::mode log, std::ostream& (*manip)(std::ostream&)) {
  if (log >= Logger::current_mode) {
    manip(std::cerr);
  }
  return log;
}

}  // namespace ale
