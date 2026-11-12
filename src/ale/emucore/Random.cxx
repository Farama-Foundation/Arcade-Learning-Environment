//============================================================================
//
//   SSSS    tt          lll  lll
//  SS  SS   tt           ll   ll
//  SS     tttttt  eeee   ll   ll   aaaa
//   SSSS    tt   ee  ee  ll   ll      aa
//      SS   tt   eeeeee  ll   ll   aaaaa  --  "An Atari 2600 VCS Emulator"
//  SS  SS   tt   ee      ll   ll  aa  aa
//   SSSS     ttt  eeeee llll llll  aaaaa
//
// Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: Random.cxx,v 1.4 2007/01/01 18:04:49 stephena Exp $
//============================================================================

#include "ale/emucore/Random.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"

#include <charconv>
#include <cstring>
#include <string>

namespace ale {
namespace stella {

namespace {

// MT19937 parameters, identical to std::mt19937.
constexpr uint32_t kN = 624;
constexpr uint32_t kM = 397;
constexpr uint32_t kMatrixA = 0x9908b0dfu;
constexpr uint32_t kUpperMask = 0x80000000u;
constexpr uint32_t kLowerMask = 0x7fffffffu;
constexpr uint32_t kDefaultSeed = 5489u;

// Saved state: this tag, then the 624 state words and the index, each as a
// little-endian uint32. The tag starts with a non-digit so it can never be
// confused with the legacy iostream text, which starts with a digit.
constexpr char kStateTag[4] = {'M', 'T', 'b', '1'};
constexpr size_t kStateSize = sizeof(kStateTag) + (kN + 1) * 4;

inline uint32_t twistWord(uint32_t upper, uint32_t lower, uint32_t shifted) {
  uint32_t y = (upper & kUpperMask) | (lower & kLowerMask);
  return shifted ^ (y >> 1) ^ ((y & 1u) ? kMatrixA : 0u);
}

}  // namespace

// Mersenne Twister with the same output as std::mt19937, but with a state
// layout we control so it serializes identically on every platform (the
// standard library's text format differs between libstdc++ and libc++).
class Random::Impl {

  public:

    Impl();

    // Implementations of the methods defined in Random.hpp.
    void seed(uint32_t value);
    uint32_t next();
    double nextDouble();

  private:

    friend class Random;

    // Regenerates the whole block of kN state words.
    void twist();

    // Restores the state from the text written by std::mt19937's operator<<.
    void loadLegacyText(const std::string& text);

    // Seed to use for creating new random number generators
    uint32_t m_seed;

    // The current block of state words, and the index of the next word to
    // temper; kN means the block is used up and must be twisted first.
    uint32_t m_state[kN];
    uint32_t m_index;
};

Random::Impl::Impl()
{
  seed(kDefaultSeed);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Random::Impl::seed(uint32_t value)
{
  m_seed = value;
  m_state[0] = value;
  for (uint32_t i = 1; i < kN; ++i)
    m_state[i] = 1812433253u * (m_state[i - 1] ^ (m_state[i - 1] >> 30)) + i;
  m_index = kN;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Random::Impl::twist()
{
  uint32_t i = 0;
  for (; i < kN - kM; ++i)
    m_state[i] = twistWord(m_state[i], m_state[i + 1], m_state[i + kM]);
  for (; i < kN - 1; ++i)
    m_state[i] = twistWord(m_state[i], m_state[i + 1], m_state[i + kM - kN]);
  m_state[kN - 1] = twistWord(m_state[kN - 1], m_state[0], m_state[kM - 1]);
  m_index = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint32_t Random::Impl::next()
{
  if (m_index >= kN)
    twist();

  uint32_t y = m_state[m_index++];
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  return y ^ (y >> 18);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
double Random::Impl::nextDouble()
{
  return next() / 4294967296.0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Random::Impl::loadLegacyText(const std::string& text)
{
  // libc++ and MSVC write the last kN outputs of the recurrence, oldest first,
  // which is the block a twist expects as input. libstdc++ writes its raw block
  // followed by its index, which matches our layout directly.
  uint32_t values[kN + 1];
  uint32_t count = 0;
  const char* p = text.data();
  const char* end = p + text.size();
  while (p < end && count <= kN) {
    while (p < end && *p == ' ')
      ++p;
    if (p == end)
      break;
    auto result = std::from_chars(p, end, values[count]);
    if (result.ec != std::errc())
      throw "Random: invalid legacy state";
    p = result.ptr;
    ++count;
  }

  if (p != end || count < kN || (count == kN + 1 && values[kN] > kN))
    throw "Random: invalid legacy state";

  std::memcpy(m_state, values, sizeof(m_state));
  m_index = count == kN + 1 ? values[kN] : kN;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Random::Random() :
    m_pimpl(new Random::Impl())
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Random::~Random() {
  if (m_pimpl != NULL) {
    delete m_pimpl;
    m_pimpl = NULL;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Random::seed(uint32_t value)
{
  m_pimpl->seed(value);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint32_t Random::next()
{
  return m_pimpl->next();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
double Random::nextDouble()
{
  return m_pimpl->nextDouble();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Random::saveState(Serializer& ser) {
  std::string buf(kStateSize, '\0');
  std::memcpy(&buf[0], kStateTag, sizeof(kStateTag));
  unsigned char* out = reinterpret_cast<unsigned char*>(&buf[sizeof(kStateTag)]);
  for (uint32_t i = 0; i <= kN; ++i) {
    uint32_t v = i < kN ? m_pimpl->m_state[i] : m_pimpl->m_index;
    out[4 * i] = v & 0xff;
    out[4 * i + 1] = (v >> 8) & 0xff;
    out[4 * i + 2] = (v >> 16) & 0xff;
    out[4 * i + 3] = (v >> 24) & 0xff;
  }

  ser.putString(buf);

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Random::loadState(Deserializer& deser) {
  std::string buf = deser.getString();

  if (buf.size() != kStateSize ||
      std::memcmp(buf.data(), kStateTag, sizeof(kStateTag)) != 0) {
    // States saved before the binary format hold std::mt19937's text.
    m_pimpl->loadLegacyText(buf);
    return true;
  }

  const unsigned char* in =
      reinterpret_cast<const unsigned char*>(buf.data() + sizeof(kStateTag));
  uint32_t values[kN + 1];
  for (uint32_t i = 0; i <= kN; ++i) {
    values[i] = uint32_t(in[4 * i]) | (uint32_t(in[4 * i + 1]) << 8) |
                (uint32_t(in[4 * i + 2]) << 16) | (uint32_t(in[4 * i + 3]) << 24);
  }
  if (values[kN] > kN)
    throw "Random: invalid state";

  std::memcpy(m_pimpl->m_state, values, sizeof(m_pimpl->m_state));
  m_pimpl->m_index = values[kN];

  return true;
}

}  // namespace stella
}  // namespace ale
