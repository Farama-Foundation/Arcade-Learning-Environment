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

// TODO(mgb): bring this include in once we switch to C++11.
// #include <random>
#include "TinyMT/tinymt32.h"

// A static Random object for compatibility purposes. Don't use this.
Random Random::s_random;

// Implementation of Random's random number generator wrapper. 
class Random::Impl {
  
  typedef tinymt32_t randgen_t;

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
    //seed((uInt32) time(NULL));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Random::Impl::seed(uInt32 value)
{
  m_seed = value;
  // TODO(mgb): this is the C++11 variant. 
  // rndGenerator.seed(ourSeed);
  tinymt32_init(&m_randgen, m_seed);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 Random::Impl::next() 
{
  // TODO(mgb): C++11
  // return rndGenerator();
  return static_cast<uInt32>(tinymt32_generate_uint32(&m_randgen));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
double Random::Impl::nextDouble()
{
  // TODO(mgb): C++11
  // return rndGenerator() / double(rndGenerator.max() + 1.0);
  return tinymt32_generate_32double(&m_randgen);
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

  // Serialize the TinyMT state
  for (int i = 0; i < 4; i++)
    ser.putInt(m_pimpl->m_randgen.status[i]);
  // These aren't really needed, but we serialize them anyway 
  ser.putInt(m_pimpl->m_randgen.mat1);
  ser.putInt(m_pimpl->m_randgen.mat2);
  ser.putInt(m_pimpl->m_randgen.tmat);

  return true;
}

bool Random::loadState(Deserializer& deser) {

  // Deserialize the TinyMT state
  for (int i = 0; i < 4; i++)
    m_pimpl->m_randgen.status[i] = deser.getInt();
  m_pimpl->m_randgen.mat1 = deser.getInt();
  m_pimpl->m_randgen.mat2 = deser.getInt();
  m_pimpl->m_randgen.tmat = deser.getInt();

  return true;
}
