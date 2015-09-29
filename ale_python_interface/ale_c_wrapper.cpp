//all code is currently in the .h file
#include "ale_c_wrapper.h"
#include<string>
#include<cstring>

const char *encodeState(ALEState *state) {
	std::string s = state->encode();
	int len = s.length();

	char *buf = (char *)malloc(len+1);
	strcpy(buf,s.c_str());

	return buf;
}

ALEState *decodeState(const char *serialized) {
	std::string s(serialized);
	return new ALEState(s);
}