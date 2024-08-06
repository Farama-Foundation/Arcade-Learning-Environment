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
// $Id: CartE0.cxx,v 1.13 2007/01/14 16:17:53 stephena Exp $
//============================================================================

#include <cassert>

#include "ale/emucore/System.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/CartE0.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeE0::CartridgeE0(const uint8_t* image)
{
  // Copy the ROM image into my buffer
  for(uint32_t addr = 0; addr < 8192; ++addr)
  {
    myImage[addr] = image[addr];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeE0::~CartridgeE0()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* CartridgeE0::name() const
{
  return "CartridgeE0";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE0::reset()
{
  // Setup segments to some default slices
  segmentZero(4);
  segmentOne(5);
  segmentTwo(6);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE0::install(System& system)
{
  mySystem = &system;
  uint16_t shift = mySystem->pageShift();
  uint16_t mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert(((0x1000 & mask) == 0) && ((0x1400 & mask) == 0) &&
      ((0x1800 & mask) == 0) && ((0x1C00 & mask) == 0));

  // Set the page acessing methods for the first part of the last segment
  System::PageAccess access;
  access.directPokeBase = 0;
  access.device = this;
  for(uint32_t i = 0x1C00; i < (0x1FE0U & ~mask); i += (1 << shift))
  {
    access.directPeekBase = &myImage[7168 + (i & 0x03FF)];
    mySystem->setPageAccess(i >> shift, access);
  }
  myCurrentSlice[3] = 7;

  // Set the page accessing methods for the hot spots in the last segment
  access.directPeekBase = 0;
  access.directPokeBase = 0;
  access.device = this;
  for(uint32_t j = (0x1FE0 & ~mask); j < 0x2000; j += (1 << shift))
  {
    mySystem->setPageAccess(j >> shift, access);
  }

  // Install some default slices for the other segments
  segmentZero(4);
  segmentOne(5);
  segmentTwo(6);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t CartridgeE0::peek(uint16_t address)
{
  address = address & 0x0FFF;

  if(!bankLocked) {
    // Switch banks if necessary
    if((address >= 0x0FE0) && (address <= 0x0FE7))
    {
      segmentZero(address & 0x0007);
    }
    else if((address >= 0x0FE8) && (address <= 0x0FEF))
    {
      segmentOne(address & 0x0007);
    }
    else if((address >= 0x0FF0) && (address <= 0x0FF7))
    {
      segmentTwo(address & 0x0007);
    }
  }

  return myImage[(myCurrentSlice[address >> 10] << 10) + (address & 0x03FF)];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE0::poke(uint16_t address, uint8_t)
{
  address = address & 0x0FFF;

  if(!bankLocked) {
    // Switch banks if necessary
    if((address >= 0x0FE0) && (address <= 0x0FE7))
    {
      segmentZero(address & 0x0007);
    }
    else if((address >= 0x0FE8) && (address <= 0x0FEF))
    {
      segmentOne(address & 0x0007);
    }
    else if((address >= 0x0FF0) && (address <= 0x0FF7))
    {
      segmentTwo(address & 0x0007);
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE0::segmentZero(uint16_t slice)
{
  // Remember the new slice
  myCurrentSlice[0] = slice;
  uint16_t offset = slice << 10;
  uint16_t shift = mySystem->pageShift();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  for(uint32_t address = 0x1000; address < 0x1400; address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x03FF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE0::segmentOne(uint16_t slice)
{
  // Remember the new slice
  myCurrentSlice[1] = slice;
  uint16_t offset = slice << 10;
  uint16_t shift = mySystem->pageShift();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  for(uint32_t address = 0x1400; address < 0x1800; address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x03FF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE0::segmentTwo(uint16_t slice)
{
  // Remember the new slice
  myCurrentSlice[2] = slice;
  uint16_t offset = slice << 10;
  uint16_t shift = mySystem->pageShift();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  for(uint32_t address = 0x1800; address < 0x1C00; address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x03FF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeE0::save(Serializer& out)
{
  std::string cart = name();

  try
  {
    out.putString(cart);

    out.putInt(4);
    for(uint32_t i = 0; i < 4; ++i)
      out.putInt(myCurrentSlice[i]);
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
bool CartridgeE0::load(Deserializer& in)
{
  std::string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    uint32_t limit = (uint32_t) in.getInt();
    for(uint32_t i = 0; i < limit; ++i)
      myCurrentSlice[i] = (uint16_t) in.getInt();
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
void CartridgeE0::bank(uint16_t bank)
{
  // FIXME - get this working, so we can debug E0 carts
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeE0::bank()
{
  // FIXME - get this working, so we can debug E0 carts
  return 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeE0::bankCount()
{
  // FIXME - get this working, so we can debug E0 carts
  return 1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeE0::patch(uint16_t address, uint8_t value)
{
  address = address & 0x0FFF;
  myImage[(myCurrentSlice[address >> 10] << 10) + (address & 0x03FF)] = value;
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t* CartridgeE0::getImage(int& size)
{
  size = 8192;
  return &myImage[0];
}

}  // namespace stella
}  // namespace ale
