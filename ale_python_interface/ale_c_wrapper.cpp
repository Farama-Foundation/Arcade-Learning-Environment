#include "ale_c_wrapper.h"
#include<string>
#include<cstring>

const char *encodeState(ALEState *state, char *buf) {
	std::string s = state->encode();
	std::strcpy(buf,s.c_str());

	return buf;
}

int encodeStateLen(ALEState *state) {
	return state->encode().length();
}

ALEState *decodeState(const char *serialized) {
	std::string s(serialized);
	return new ALEState(s);
}