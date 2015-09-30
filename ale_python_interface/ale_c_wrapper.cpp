#include "ale_c_wrapper.h"
#include<vector>
#include<cstring>

void encodeState(ALEState *state, char *buf) {
	std::vector<char> vec = state->encode();
	memcpy(buf,vec.data(),vec.size());
}

int encodeStateLen(ALEState *state) {
	return state->encode().size();
}

ALEState *decodeState(const char *serialized, int len) {
	std::vector<char> buf;
	buf.resize(len);

	memcpy(buf.data(), serialized, len);
	return new ALEState(buf);
}