#include "ale_c_wrapper.h"

#include <cstring>
#include <string>
#include <stdexcept>

void encodeState(ale::ALEState* state, char* buf, int buf_len) {
  std::string str = state->serialize();

  if (buf_len < int(str.length())) {
    throw new std::runtime_error(
        "Buffer is not big enough to hold serialized ale::ALEState. Please use "
        "encodeStateLen to determine the correct buffer size");
  }

  std::memcpy(buf, str.data(), str.length());
}

int encodeStateLen(ale::ALEState* state) { return state->serialize().length(); }

ale::ALEState* decodeState(const char* serialized, int len) {
  std::string str(serialized, len);

  return new ale::ALEState(str);
}
