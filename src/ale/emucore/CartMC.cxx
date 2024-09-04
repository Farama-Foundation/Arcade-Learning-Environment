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
// $Id: CartMC.cxx,v 1.12 2007/01/14 16:17:55 stephena Exp $
//============================================================================

#include <cassert>

#include "ale/emucore/System.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/CartMC.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeMC::CartridgeMC(const uint8_t* image, uint32_t size)
  : mySlot3Locked(false)
{
  // Make sure size is reasonable
  assert(size <= 128 * 1024);

  // Allocate array for the cart's RAM
  myRAM = new uint8_t[32 * 1024];

  // Allocate array for the ROM image
  myImage = new uint8_t[128 * 1024];

  // Set the contents of the entire ROM to 0
  for(uint32_t i = 0; i < 128 * 1024; ++i)
  {
    myImage[i] = 0;
  }

  // Copy the ROM image to the end of the ROM buffer
  for(uint32_t i = 0; i < size; ++i)
  {
    myImage[128 * 1024 - size + i] = image[i];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeMC::~CartridgeMC()
{
  delete[] myRAM;
  delete[] myImage;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* CartridgeMC::name() const
{
  return "CartridgeMC";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeMC::reset()
{
  // Initialize RAM with random values
  for(uint32_t i = 0; i < 32 * 1024; ++i)
    myRAM[i] = mySystem->rng().next();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeMC::install(System& system)
{
  mySystem = &system;
  uint16_t shift = mySystem->pageShift();
  uint16_t mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert(((0x1000 & mask) == 0) && ((0x1400 & mask) == 0) &&
      ((0x1800 & mask) == 0) && ((0x1C00 & mask) == 0));

  // Set the page accessing methods for the hot spots in the TIA.  For
  // correct emulation I would need to chain any accesses below 0x40 to
  // the TIA but for now I'll just forget about them.
  //
  // TODO: These TIA accesses may need to be chained, however, at this
  //       point Chris isn't sure if the hardware will allow it or not
  //
  System::PageAccess access;
  for(uint32_t i = 0x00; i < 0x40; i += (1 << shift))
  {
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    access.device = this;
    mySystem->setPageAccess(i >> shift, access);
  }

  // Map the cartridge into the system
  for(uint32_t j = 0x1000; j < 0x2000; j += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    mySystem->setPageAccess(j >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t CartridgeMC::peek(uint16_t address)
{
  address = address & 0x1FFF;

  // Accessing the RESET vector so lets handle the powerup special case
  if((address == 0x1FFC) || (address == 0x1FFD))
  {
    // Indicate that slot 3 is locked for now
    mySlot3Locked = true;
  }
  // Should we unlock slot 3?
  else if(mySlot3Locked && (address >= 0x1000) && (address <= 0x1BFF))
  {
    // Indicate that slot 3 is unlocked now
    mySlot3Locked = false;
  }

  // Handle reads made to the TIA addresses
  if(address < 0x1000)
  {
    return 0;
  }
  else
  {
    uint8_t block;

    if(mySlot3Locked && ((address & 0x0C00) == 0x0C00))
    {
      block = 0xFF;
    }
    else
    {
      block = myCurrentBlock[(address & 0x0C00) >> 10];
    }

    // Is this a RAM or a ROM access
    if(block & 0x80)
    {
      // ROM access
      return myImage[(uint32_t)(block & 0x7F) * 1024 + (address & 0x03FF)];
    }
    else
    {
      // This is a RAM access, however, is it to the read or write port?
      if(address & 0x0200)
      {
        // Reading from the read port of the RAM block
        return myRAM[(uint32_t)(block & 0x3F) * 512 + (address & 0x01FF)];
      }
      else
      {
        // Oops, reading from the write port of the RAM block!
        myRAM[(uint32_t)(block & 0x3F) * 512 + (address & 0x01FF)] = 0;
        return 0;
      }
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeMC::poke(uint16_t address, uint8_t value)
{
  address = address & 0x1FFF;

  // Accessing the RESET vector so lets handle the powerup special case
  if((address == 0x1FFC) || (address == 0x1FFD))
  {
    // Indicate that slot 3 is locked for now
    mySlot3Locked = true;
  }
  // Should we unlock slot 3?
  else if(mySlot3Locked && (address >= 0x1000) && (address <= 0x1BFF))
  {
    // Indicate that slot 3 is unlocked now
    mySlot3Locked = false;
  }

  // Handle bank-switching writes
  if((address >= 0x003C) && (address <= 0x003F))
  {
    myCurrentBlock[address - 0x003C] = value;
  }
  else
  {
    uint8_t block;

    if(mySlot3Locked && ((address & 0x0C00) == 0x0C00))
    {
      block = 0xFF;
    }
    else
    {
      block = myCurrentBlock[(address & 0x0C00) >> 10];
    }

    // Is this a RAM write access
    if(!(block & 0x80) && !(address & 0x0200))
    {
      // Handle the write to RAM
      myRAM[(uint32_t)(block & 0x3F) * 512 + (address & 0x01FF)] = value;
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeMC::save(Serializer& out)
{
  uint32_t i;
  std::string cart = name();

  try
  {
    out.putString(cart);

    // The currentBlock array
    out.putInt(4);
    for(i = 0; i < 4; ++i)
      out.putInt(myCurrentBlock[i]);

    // The 32K of RAM
    out.putInt(32 * 1024);
    for(i = 0; i < 32 * 1024; ++i)
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
bool CartridgeMC::load(Deserializer& in)
{
  uint32_t i;
  std::string cart = name();

  try
  {
    uint32_t limit;

    if(in.getString() != cart)
      return false;

    // The currentBlock array
    limit = (uint32_t) in.getInt();
    for(i = 0; i < limit; ++i)
      myCurrentBlock[i] = (uint8_t) in.getInt();

    // The 32K of RAM
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

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeMC::bank(uint16_t b)
{
  // TODO: add support for debugger
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeMC::bank()
{
  // TODO: add support for debugger
  return 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeMC::bankCount()
{
  // TODO: add support for debugger
  return 1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeMC::patch(uint16_t address, uint8_t value)
{
  // TODO: implement
  return false;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t* CartridgeMC::getImage(int& size)
{
  size = 128 * 1024; // FIXME: keep track of original size
  return &myImage[0];
}

}  // namespace stella
}  // namespace ale
