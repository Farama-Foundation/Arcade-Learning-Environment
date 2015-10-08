#include "ale_c_wrapper.h"

#include <cstring>
#include <string>
#include <stdexcept>

void encodeState(ALEState *state, char *buf, int buf_len) {
	std::string str = state->serialize();

	if (buf_len < int(str.length())) {
		throw new std::runtime_error("Buffer is not big enough to hold serialized ALEState. Please use encodeStateLen to determine the correct buffer size");
	}

	memcpy(buf, str.data(), str.length());
}

int encodeStateLen(ALEState *state) {
	return state->serialize().length();
}

ALEState *decodeState(const char *serialized, int len) {
	std::string str(serialized, len);

	return new ALEState(str);
}