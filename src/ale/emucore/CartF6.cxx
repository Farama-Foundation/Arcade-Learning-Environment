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
// $Id: CartF6.cxx,v 1.13 2007/01/14 16:17:54 stephena Exp $
//============================================================================

#include <cassert>

#include "ale/emucore/System.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/CartF6.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeF6::CartridgeF6(const uint8_t* image)
{
  // Copy the ROM image into my buffer
  for(uint32_t addr = 0; addr < 16384; ++addr)
  {
    myImage[addr] = image[addr];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeF6::~CartridgeF6()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* CartridgeF6::name() const
{
  return "CartridgeF6";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF6::reset()
{
  // Upon reset we switch to bank 0
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF6::install(System& system)
{
  mySystem = &system;
  uint16_t shift = mySystem->pageShift();
  uint16_t mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert((0x1000 & mask) == 0);

  // Set the page accessing methods for the hot spots
  System::PageAccess access;
  for(uint32_t i = (0x1FF6 & ~mask); i < 0x2000; i += (1 << shift))
  {
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    access.device = this;
    mySystem->setPageAccess(i >> shift, access);
  }

  // Upon install we'll setup bank 0
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t CartridgeF6::peek(uint16_t address)
{
  address = address & 0x0FFF;

  // Switch banks if necessary
  switch(address)
  {
    case 0x0FF6:
      // Set the current bank to the first 4k bank
      bank(0);
      break;

    case 0x0FF7:
      // Set the current bank to the second 4k bank
      bank(1);
      break;

    case 0x0FF8:
      // Set the current bank to the third 4k bank
      bank(2);
      break;

    case 0x0FF9:
      // Set the current bank to the forth 4k bank
      bank(3);
      break;

    default:
      break;
  }

  return myImage[myCurrentBank * 4096 + address];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF6::poke(uint16_t address, uint8_t)
{
  address = address & 0x0FFF;

  // Switch banks if necessary
  switch(address)
  {
    case 0x0FF6:
      // Set the current bank to the first 4k bank
      bank(0);
      break;

    case 0x0FF7:
      // Set the current bank to the second 4k bank
      bank(1);
      break;

    case 0x0FF8:
      // Set the current bank to the third 4k bank
      bank(2);
      break;

    case 0x0FF9:
      // Set the current bank to the forth 4k bank
      bank(3);
      break;

    default:
      break;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeF6::save(Serializer& out)
{
  std::string cart = name();

  try
  {
    out.putString(cart);

    out.putInt(myCurrentBank);
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
bool CartridgeF6::load(Deserializer& in)
{
  std::string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    myCurrentBank = (uint16_t) in.getInt();
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

  // Remember what bank we were in
  bank(myCurrentBank);

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF6::bank(uint16_t bank)
{
  if(bankLocked) return;

  // Remember what bank we're in
  myCurrentBank = bank;
  uint16_t offset = myCurrentBank * 4096;
  uint16_t shift = mySystem->pageShift();
  uint16_t mask = mySystem->pageMask();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  // Map ROM image into the system
  for(uint32_t address = 0x1000; address < (0x1FF6U & ~mask);
      address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x0FFF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeF6::bank()
{
  return myCurrentBank;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeF6::bankCount()
{
  return 4;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeF6::patch(uint16_t address, uint8_t value)
{
  address = address & 0x0FFF;
  myImage[myCurrentBank * 4096 + address] = value;
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t* CartridgeF6::getImage(int& size)
{
  size = 16384;
  return &myImage[0];
}

}  // namespace stella
}  // namespace ale
