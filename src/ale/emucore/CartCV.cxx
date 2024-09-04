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
// $Id: CartCV.cxx,v 1.14 2007/01/14 16:17:53 stephena Exp $
//============================================================================

#include <cassert>
#include <cstring>

#include "ale/emucore/System.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/CartCV.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeCV::CartridgeCV(const uint8_t* image, uint32_t size)
{
  uint32_t addr;
  if(size == 2048)
  {
    // Copy the ROM image into my buffer
    for(uint32_t addr = 0; addr < 2048; ++addr)
    {
      myImage[addr] = image[addr];
    }
  }
  else if(size == 4096)
  {
    // The game has something saved in the RAM
    // Usefull for MagiCard program listings

    // Copy the ROM image into my buffer
    for(addr = 0; addr < 2048; ++addr)
    {
      myImage[addr] = image[addr + 2048];
    }

    myInitialRAM = new uint8_t[1024];
    std::memcpy(myInitialRAM, image, 1024);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeCV::~CartridgeCV()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* CartridgeCV::name() const
{
  return "CartridgeCV";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeCV::reset()
{
  if (myInitialRAM) {
    // Copy the RAM image into my buffer
    std::memcpy(myRAM, myInitialRAM, 1024);
  } else {
    // Initialize RAM with random values
    for(uint32_t i = 0; i < 1024; ++i)
      myRAM[i] = mySystem->rng().next();
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeCV::install(System& system)
{
  mySystem = &system;
  uint16_t shift = mySystem->pageShift();
  uint16_t mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert((0x1800 & mask) == 0);

  System::PageAccess access;
  access.directPokeBase = 0;
  access.device = this;

  // Map ROM image into the system
  for(uint32_t address = 0x1800; address < 0x2000; address += (1 << shift))
  {
    access.directPeekBase = &myImage[address & 0x07FF];
    mySystem->setPageAccess(address >> mySystem->pageShift(), access);
  }

  // Set the page accessing method for the RAM writing pages
  for(uint32_t j = 0x1400; j < 0x1800; j += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = 0;
    access.directPokeBase = &myRAM[j & 0x03FF];
    mySystem->setPageAccess(j >> shift, access);
  }

  // Set the page accessing method for the RAM reading pages
  for(uint32_t k = 0x1000; k < 0x1400; k += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = &myRAM[k & 0x03FF];
    access.directPokeBase = 0;
    mySystem->setPageAccess(k >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t CartridgeCV::peek(uint16_t address)
{
  return myImage[address & 0x07FF];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeCV::poke(uint16_t, uint8_t)
{
  // This is ROM so poking has no effect :-)
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeCV::save(Serializer& out)
{
  std::string cart = name();

  try
  {
    out.putString(cart);

    // Output RAM
    out.putInt(1024);
    for(uint32_t addr = 0; addr < 1024; ++addr)
      out.putInt(myRAM[addr]);
  }
  catch(const char* msg)
  {
    ale::Logger::Error << msg << std::endl;
    return false;
  }
  catch(...)
  {
    ale::Logger::Error << "Unknown error in save state for " << cart << std::endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeCV::load(Deserializer& in)
{
  std::string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    // Input RAM
    uint32_t limit = (uint32_t) in.getInt();
    for(uint32_t addr = 0; addr < limit; ++addr)
      myRAM[addr] = (uint8_t) in.getInt();
  }
  catch(const char* msg)
  {
    ale::Logger::Error << msg << std::endl;
    return false;
  }
  catch(...)
  {
    ale::Logger::Error << "Unknown error in load state for " << cart << std::endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeCV::bank(uint16_t bank)
{
  // Doesn't support bankswitching
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeCV::bank()
{
  // Doesn't support bankswitching
  return 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeCV::bankCount()
{
  return 1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeCV::patch(uint16_t address, uint8_t value)
{
  myImage[address & 0x07FF] = value;
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t* CartridgeCV::getImage(int& size)
{
  size = 2048;
  return &myImage[0];
}

}  // namespace stella
}  // namespace ale
