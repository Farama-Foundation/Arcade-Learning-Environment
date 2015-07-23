#ifndef LOG_H
#define LOG_H
#include <iostream>
namespace ale
{
    class Logger
    {
    public:
        enum mode{
            Info = 0,
            Warning = 1,
            Error = 2
        };
        static void setMode(mode m);
        static mode current_mode;
    };


    Logger::mode operator<<(Logger::mode log, std::ostream & (*manip)(std::ostream &));
    
    template<typename T>
    Logger::mode operator << (Logger::mode log, const T& val){
        if(log>=Logger::current_mode)
            std::cerr << val;
        return log;
    }


}
#endif
