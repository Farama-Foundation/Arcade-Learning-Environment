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

#include <time.h>
#include "Random.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"

// This uses C++11.
#include <random>
#include <sstream>

// A static Random object for compatibility purposes. Don't use this.
Random Random::s_random;

// Implementation of Random's random number generator wrapper. 
class Random::Impl {
  
  typedef std::mt19937 randgen_t;

  public:
    
    Impl();

    // Implementations of the methods defined in Random.hpp.
    void seed(uInt32 value);
    uInt32 next();
    double nextDouble();

  private:
   
    friend class Random;

    // Seed to use for creating new random number generators
    uInt32 m_seed;

    // Random number generator 
    randgen_t m_randgen; 
};

Random::Impl::Impl()
{
    // Initialize seed to time
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Random::Impl::seed(uInt32 value)
{
  m_seed = value;
  m_randgen.seed(m_seed);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 Random::Impl::next() 
{
  return m_randgen();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
double Random::Impl::nextDouble()
{
  return m_randgen() / double(m_randgen.max() + 1.0);
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
void Random::seed(uInt32 value)
{
  m_pimpl->seed(value);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 Random::next()
{
  return m_pimpl->next();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
double Random::nextDouble()
{
  return m_pimpl->nextDouble();
}

Random& Random::getInstance() {
  return s_random;
}

bool Random::saveState(Serializer& ser) {
  // The mt19937 object's serialization of choice is into a string. 
  std::ostringstream oss;
  oss << m_pimpl->m_randgen;

  ser.putString(oss.str());

  return true;
}

bool Random::loadState(Deserializer& deser) {
  // Deserialize into a string.
  std::istringstream iss(deser.getString());

  iss >> m_pimpl->m_randgen;

  return true;
}
