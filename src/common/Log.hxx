#ifndef LOG_H
#define LOG_H

#ifdef VERBOSE
#define LOG(msg) do { std::cerr << msg; } while(false)
#else
#define LOG(msg) do { } while(false)
#endif

#endif
