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
// $Id: CartE7.cxx,v 1.16 2007/01/14 16:17:53 stephena Exp $
//============================================================================

#include <cassert>

#include "ale/emucore/System.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/CartE7.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeE7::CartridgeE7(const uint8_t* image)
{
  // Copy the ROM image into my buffer
  for(uint32_t addr = 0; addr < 16384; ++addr)
  {
    myImage[addr] = image[addr];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeE7::~CartridgeE7()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* CartridgeE7::name() const
{
  return "CartridgeE7";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE7::reset()
{
  // Initialize RAM with random values
  for(uint32_t i = 0; i < 2048; ++i)
    myRAM[i] = mySystem->rng().next();

  // Install some default banks for the RAM and first segment
  bankRAM(0);
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE7::install(System& system)
{
  mySystem = &system;
  uint16_t shift = mySystem->pageShift();
  uint16_t mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert(((0x1400 & mask) == 0) && ((0x1800 & mask) == 0) &&
      ((0x1900 & mask) == 0) && ((0x1A00 & mask) == 0));

  // Set the page accessing methods for the hot spots
  System::PageAccess access;
  for(uint32_t i = (0x1FE0 & ~mask); i < 0x2000; i += (1 << shift))
  {
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    access.device = this;
    mySystem->setPageAccess(i >> shift, access);
  }

  // Setup the second segment to always point to the last ROM slice
  for(uint32_t j = 0x1A00; j < (0x1FE0U & ~mask); j += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = &myImage[7 * 2048 + (j & 0x07FF)];
    access.directPokeBase = 0;
    mySystem->setPageAccess(j >> shift, access);
  }
  myCurrentSlice[1] = 7;

  // Install some default banks for the RAM and first segment
  bankRAM(0);
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t CartridgeE7::peek(uint16_t address)
{
  address = address & 0x0FFF;

  // Switch banks if necessary
  if((address >= 0x0FE0) && (address <= 0x0FE7))
  {
    bank(address & 0x0007);
  }
  else if((address >= 0x0FE8) && (address <= 0x0FEB))
  {
    bankRAM(address & 0x0003);
  }

  // NOTE: The following does not handle reading from RAM, however,
  // this function should never be called for RAM because of the
  // way page accessing has been setup
  return myImage[(myCurrentSlice[address >> 11] << 11) + (address & 0x07FF)];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE7::poke(uint16_t address, uint8_t)
{
  address = address & 0x0FFF;

  // Switch banks if necessary
  if((address >= 0x0FE0) && (address <= 0x0FE7))
  {
    bank(address & 0x0007);
  }
  else if((address >= 0x0FE8) && (address <= 0x0FEB))
  {
    bankRAM(address & 0x0003);
  }

  // NOTE: This does not handle writing to RAM, however, this
  // function should never be called for RAM because of the
  // way page accessing has been setup
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE7::bankRAM(uint16_t bank)
{
  // Remember what bank we're in
  myCurrentRAM = bank;
  uint16_t offset = bank << 8;
  uint16_t shift = mySystem->pageShift();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;

  // Set the page accessing method for the 256 bytes of RAM writing pages
  access.directPeekBase = 0;
  access.directPokeBase = 0;
  for(uint32_t j = 0x1800; j < 0x1900; j += (1 << shift))
  {
    access.directPokeBase = &myRAM[1024 + offset + (j & 0x00FF)];
    mySystem->setPageAccess(j >> shift, access);
  }

  // Set the page accessing method for the 256 bytes of RAM reading pages
  access.directPeekBase = 0;
  access.directPokeBase = 0;
  for(uint32_t k = 0x1900; k < 0x1A00; k += (1 << shift))
  {
    access.directPeekBase = &myRAM[1024 + offset + (k & 0x00FF)];
    mySystem->setPageAccess(k >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeE7::save(Serializer& out)
{
  std::string cart = name();

  try
  {
    uint32_t i;

    out.putString(cart);

    out.putInt(2);
    for(i = 0; i < 2; ++i)
      out.putInt(myCurrentSlice[i]);

    out.putInt(myCurrentRAM);

    // The 2048 bytes of RAM
    out.putInt(2048);
    for(i = 0; i < 2048; ++i)
      out.putInt(myRAM[i]);
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
bool CartridgeE7::load(Deserializer& in)
{
  std::string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    uint32_t i, limit;

    limit = (uint32_t) in.getInt();
    for(i = 0; i < limit; ++i)
      myCurrentSlice[i] = (uint16_t) in.getInt();

    myCurrentRAM = (uint16_t) in.getInt();

    // The 2048 bytes of RAM
    limit = (uint32_t) in.getInt();
    for(i = 0; i < limit; ++i)
      myRAM[i] = (uint8_t) in.getInt();
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

  // Set up the previously used banks for the RAM and segment
  bankRAM(myCurrentRAM);
  bank(myCurrentSlice[0]);

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE7::bank(uint16_t slice)
{
  if(bankLocked) return;

  // Remember what bank we're in
  myCurrentSlice[0] = slice;
  uint16_t offset = slice << 11;
  uint16_t shift = mySystem->pageShift();

  // Setup the page access methods for the current bank
  if(slice != 7)
  {
    System::PageAccess access;
    access.device = this;
    access.directPokeBase = 0;

    // Map ROM image into first segment
    for(uint32_t address = 0x1000; address < 0x1800; address += (1 << shift))
    {
      access.directPeekBase = &myImage[offset + (address & 0x07FF)];
      mySystem->setPageAccess(address >> shift, access);
    }
  }
  else
  {
    System::PageAccess access;
    access.device = this;

    // Set the page accessing method for the 1K slice of RAM writing pages
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    for(uint32_t j = 0x1000; j < 0x1400; j += (1 << shift))
    {
      access.directPokeBase = &myRAM[j & 0x03FF];
      mySystem->setPageAccess(j >> shift, access);
    }

    // Set the page accessing method for the 1K slice of RAM reading pages
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    for(uint32_t k = 0x1400; k < 0x1800; k += (1 << shift))
    {
      access.directPeekBase = &myRAM[k & 0x03FF];
      mySystem->setPageAccess(k >> shift, access);
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeE7::bank()
{
  return myCurrentSlice[0];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeE7::bankCount()
{
  return 8;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeE7::patch(uint16_t address, uint8_t value)
{
  address = address & 0x0FFF;
  myImage[(myCurrentSlice[address >> 11] << 11) + (address & 0x07FF)] = value;
  bank(myCurrentSlice[0]);
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t* CartridgeE7::getImage(int& size)
{
  size = 16384;
  return &myImage[0];
}

}  // namespace stella
}  // namespace ale
