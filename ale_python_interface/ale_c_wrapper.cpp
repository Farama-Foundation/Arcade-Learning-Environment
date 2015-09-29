#include "ale_c_wrapper.h"
#include<string>
#include<cstring>

const char *encodeState(ALEState *state) {
	std::string s = state->encode();
	char *buf = new char[s.length() + 1];
	std::strcpy(buf,s.c_str());

	return buf;
}

ALEState *decodeState(const char *serialized) {
	std::string s(serialized);
	return new ALEState(s);
}