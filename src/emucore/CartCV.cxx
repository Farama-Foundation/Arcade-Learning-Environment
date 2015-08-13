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

#include "Random.hxx"
#include "System.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"
#include "CartCV.hxx"
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeCV::CartridgeCV(const uInt8* image, uInt32 size)
{
  uInt32 addr;
  if(size == 2048)
  {
    // Copy the ROM image into my buffer
    for(uInt32 addr = 0; addr < 2048; ++addr)
    {
      myImage[addr] = image[addr];
    }

    // Initialize RAM with random values
    class Random random;
    for(uInt32 i = 0; i < 1024; ++i)
    {
      myRAM[i] = random.next();
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

    // Copy the RAM image into my buffer
    for(addr = 0; addr < 1024; ++addr)
    {
      myRAM[addr] = image[addr];
    }

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
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeCV::install(System& system)
{
  mySystem = &system;
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert((0x1800 & mask) == 0);

  System::PageAccess access;
  access.directPokeBase = 0;
  access.device = this;

  // Map ROM image into the system
  for(uInt32 address = 0x1800; address < 0x2000; address += (1 << shift))
  {
    access.directPeekBase = &myImage[address & 0x07FF];
    mySystem->setPageAccess(address >> mySystem->pageShift(), access);
  }

  // Set the page accessing method for the RAM writing pages
  for(uInt32 j = 0x1400; j < 0x1800; j += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = 0;
    access.directPokeBase = &myRAM[j & 0x03FF];
    mySystem->setPageAccess(j >> shift, access);
  }

  // Set the page accessing method for the RAM reading pages
  for(uInt32 k = 0x1000; k < 0x1400; k += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = &myRAM[k & 0x03FF];
    access.directPokeBase = 0;
    mySystem->setPageAccess(k >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8 CartridgeCV::peek(uInt16 address)
{
  return myImage[address & 0x07FF];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeCV::poke(uInt16, uInt8)
{
  // This is ROM so poking has no effect :-)
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeCV::save(Serializer& out)
{
  string cart = name();

  try
  {
    out.putString(cart);

    // Output RAM
    out.putInt(1024);
    for(uInt32 addr = 0; addr < 1024; ++addr)
      out.putInt(myRAM[addr]);
  }
  catch(const char* msg)
  {
    ale::Logger::Error << msg << endl;
    return false;
  }
  catch(...)
  {
    ale::Logger::Error << "Unknown error in save state for " << cart << endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeCV::load(Deserializer& in)
{
  string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    // Input RAM
    uInt32 limit = (uInt32) in.getInt();
    for(uInt32 addr = 0; addr < limit; ++addr)
      myRAM[addr] = (uInt8) in.getInt();
  }
  catch(const char* msg)
  {
    ale::Logger::Error << msg << endl;
    return false;
  }
  catch(...)
  {
    ale::Logger::Error << "Unknown error in load state for " << cart << endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeCV::bank(uInt16 bank)
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
bool CartridgeCV::patch(uInt16 address, uInt8 value)
{
  myImage[address & 0x07FF] = value;
  return true;
} 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8* CartridgeCV::getImage(int& size)
{
  size = 2048;
  return &myImage[0];
}
