#include "Log.hpp"
#include <iostream>
using namespace ale;

Logger::mode Logger::current_mode = Info;

void Logger::setMode(Logger::mode m){
    current_mode = m;
}

ale::Logger::mode ale::operator<<(ale::Logger::mode log, std::ostream & (*manip)(std::ostream &)) {
    if(log >= Logger::current_mode)
        manip(std::cerr);
    return log;
}
