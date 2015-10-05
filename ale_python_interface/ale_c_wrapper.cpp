#include "ale_c_wrapper.h"

#include <cstring>
#include <string>

void encodeState(ALEState *state, char *buf) {
	std::string str = state->serialize();

	memcpy(buf, str.data(), str.length());
}

int encodeStateLen(ALEState *state) {
	return state->serialize().length();
}

ALEState *decodeState(const char *serialized, int len) {
	std::string str(serialized, len);

	return new ALEState(str);
}