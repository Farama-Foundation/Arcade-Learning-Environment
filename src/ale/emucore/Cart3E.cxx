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
// $Id: Cart3E.cxx,v 1.12 2007/01/14 16:17:52 stephena Exp $
//============================================================================

#include <cassert>

#include "ale/emucore/System.hxx"
#include "ale/emucore/TIA.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/Cart3E.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge3E::Cartridge3E(const uint8_t* image, uint32_t size)
  : mySize(size)
{
  // Allocate array for the ROM image
  myImage = new uint8_t[mySize];

  // Copy the ROM image into my buffer
  for(uint32_t addr = 0; addr < mySize; ++addr)
  {
    myImage[addr] = image[addr];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge3E::~Cartridge3E()
{
  delete[] myImage;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* Cartridge3E::name() const
{
  return "Cartridge3E";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3E::reset()
{
  // Initialize RAM with random values
  for(uint32_t i = 0; i < 32768; ++i)
    myRam[i] = mySystem->rng().next();

  // We'll map bank 0 into the first segment upon reset
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3E::install(System& system)
{
  mySystem = &system;
  uint16_t shift = mySystem->pageShift();
  uint16_t mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert((0x1800 & mask) == 0);

  // Set the page accessing methods for the hot spots (for 100% emulation
  // I would need to chain any accesses below 0x40 to the TIA but for
  // now I'll just forget about them)
  System::PageAccess access;
  for(uint32_t i = 0x00; i < 0x40; i += (1 << shift))
  {
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    access.device = this;
    mySystem->setPageAccess(i >> shift, access);
  }

  // Setup the second segment to always point to the last ROM slice
  for(uint32_t j = 0x1800; j < 0x2000; j += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = &myImage[(mySize - 2048) + (j & 0x07FF)];
    access.directPokeBase = 0;
    mySystem->setPageAccess(j >> shift, access);
  }

  // Install pages for bank 0 into the first segment
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t Cartridge3E::peek(uint16_t address)
{
  address = address & 0x0FFF;

  if(address < 0x0800)
  {
    if(myCurrentBank < 256)
      return myImage[(address & 0x07FF) + myCurrentBank * 2048];
    else
      return myRam[(address & 0x03FF) + (myCurrentBank - 256) * 1024];
  }
  else
  {
    return myImage[(address & 0x07FF) + mySize - 2048];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3E::poke(uint16_t address, uint8_t value)
{
  address = address & 0x0FFF;

  // Switch banks if necessary. Armin (Kroko) says there are no mirrored
  // hotspots.
  if(address == 0x003F)
  {
    bank(value);
  }
  else if(address == 0x003E)
  {
    bank(value + 256);
  }

  // Pass the poke through to the TIA. In a real Atari, both the cart and the
  // TIA see the address lines, and both react accordingly. In Stella, each
  // 64-byte chunk of address space is "owned" by only one device. If we
  // don't chain the poke to the TIA, then the TIA can't see it...
  mySystem->tia().poke(address, value);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge3E::save(Serializer& out)
{
  std::string cart = name();

  try
  {
    out.putString(cart);
    out.putInt(myCurrentBank);

    // Output RAM
    out.putInt(32768);
    for(uint32_t addr = 0; addr < 32768; ++addr)
      out.putInt(myRam[addr]);
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
bool Cartridge3E::load(Deserializer& in)
{
  std::string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    myCurrentBank = (uint16_t) in.getInt();

    // Input RAM
    uint32_t limit = (uint32_t) in.getInt();
    for(uint32_t addr = 0; addr < limit; ++addr)
      myRam[addr] = (uint8_t) in.getInt();
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

  // Now, go to the current bank
  bank(myCurrentBank);

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3E::bank(uint16_t bank)
{
  if(bankLocked) return;

  if(bank < 256)
  {
    // Make sure the bank they're asking for is reasonable
    if((uint32_t)bank * 2048 < mySize)
    {
      myCurrentBank = bank;
    }
    else
    {
      // Oops, the bank they're asking for isn't valid so let's wrap it
      // around to a valid bank number
      myCurrentBank = bank % (mySize / 2048);
    }

    uint32_t offset = myCurrentBank * 2048;
    uint16_t shift = mySystem->pageShift();

    // Setup the page access methods for the current bank
    System::PageAccess access;
    access.device = this;
    access.directPokeBase = 0;

    // Map ROM image into the system
    for(uint32_t address = 0x1000; address < 0x1800; address += (1 << shift))
    {
      access.directPeekBase = &myImage[offset + (address & 0x07FF)];
      mySystem->setPageAccess(address >> shift, access);
    }
  }
  else
  {
    bank -= 256;
    bank %= 32;
    myCurrentBank = bank + 256;

    uint32_t offset = bank * 1024;
    uint16_t shift = mySystem->pageShift();
    uint32_t address;

    // Setup the page access methods for the current bank
    System::PageAccess access;
    access.device = this;
    access.directPokeBase = 0;

    // Map read-port RAM image into the system
    for(address = 0x1000; address < 0x1400; address += (1 << shift))
    {
      access.directPeekBase = &myRam[offset + (address & 0x03FF)];
      mySystem->setPageAccess(address >> shift, access);
    }

    access.directPeekBase = 0;

    // Map write-port RAM image into the system
    for(address = 0x1400; address < 0x1800; address += (1 << shift))
    {
      access.directPokeBase = &myRam[offset + (address & 0x03FF)];
      mySystem->setPageAccess(address >> shift, access);
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Cartridge3E::bank()
{
  return myCurrentBank;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Cartridge3E::bankCount()
{
  return mySize / 2048;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge3E::patch(uint16_t address, uint8_t value)
{
  address = address & 0x0FFF;
  if(address < 0x0800)
  {
    if(myCurrentBank < 256)
      myImage[(address & 0x07FF) + myCurrentBank * 2048] = value;
    else
      myRam[(address & 0x03FF) + (myCurrentBank - 256) * 1024] = value;
  }
  else
  {
    myImage[(address & 0x07FF) + mySize - 2048] = value;
  }
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t* Cartridge3E::getImage(int& size)
{
  size = mySize;
  return &myImage[0];
}

}  // namespace stella
}  // namespace ale
