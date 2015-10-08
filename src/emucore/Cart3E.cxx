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

#include "Random.hxx"
#include "System.hxx"
#include "TIA.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"
#include "Cart3E.hxx"
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge3E::Cartridge3E(const uInt8* image, uInt32 size)
  : mySize(size)
{
  // Allocate array for the ROM image
  myImage = new uInt8[mySize];

  // Copy the ROM image into my buffer
  for(uInt32 addr = 0; addr < mySize; ++addr)
  {
    myImage[addr] = image[addr];
  }

  // Initialize RAM with random values
  class Random& random = Random::getInstance();
  for(uInt32 i = 0; i < 32768; ++i)
  {
    myRam[i] = random.next();
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
  // We'll map bank 0 into the first segment upon reset
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3E::install(System& system)
{
  mySystem = &system;
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert((0x1800 & mask) == 0);

  // Set the page accessing methods for the hot spots (for 100% emulation
  // I would need to chain any accesses below 0x40 to the TIA but for
  // now I'll just forget about them)
  System::PageAccess access;
  for(uInt32 i = 0x00; i < 0x40; i += (1 << shift))
  {
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    access.device = this;
    mySystem->setPageAccess(i >> shift, access);
  }

  // Setup the second segment to always point to the last ROM slice
  for(uInt32 j = 0x1800; j < 0x2000; j += (1 << shift))
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
uInt8 Cartridge3E::peek(uInt16 address)
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
void Cartridge3E::poke(uInt16 address, uInt8 value)
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
  string cart = name();

  try
  {
    out.putString(cart);
    out.putInt(myCurrentBank);

    // Output RAM
    out.putInt(32768);
    for(uInt32 addr = 0; addr < 32768; ++addr)
      out.putInt(myRam[addr]);
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
bool Cartridge3E::load(Deserializer& in)
{
  string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    myCurrentBank = (uInt16) in.getInt();

    // Input RAM
    uInt32 limit = (uInt32) in.getInt();
    for(uInt32 addr = 0; addr < limit; ++addr)
      myRam[addr] = (uInt8) in.getInt();
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

  // Now, go to the current bank
  bank(myCurrentBank);

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3E::bank(uInt16 bank)
{ 
  if(bankLocked) return;

  if(bank < 256)
  {
    // Make sure the bank they're asking for is reasonable
    if((uInt32)bank * 2048 < mySize)
    {
      myCurrentBank = bank;
    }
    else
    {
      // Oops, the bank they're asking for isn't valid so let's wrap it
      // around to a valid bank number
      myCurrentBank = bank % (mySize / 2048);
    }
  
    uInt32 offset = myCurrentBank * 2048;
    uInt16 shift = mySystem->pageShift();
  
    // Setup the page access methods for the current bank
    System::PageAccess access;
    access.device = this;
    access.directPokeBase = 0;
  
    // Map ROM image into the system
    for(uInt32 address = 0x1000; address < 0x1800; address += (1 << shift))
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

    uInt32 offset = bank * 1024;
    uInt16 shift = mySystem->pageShift();
    uInt32 address;
  
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
bool Cartridge3E::patch(uInt16 address, uInt8 value)
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
uInt8* Cartridge3E::getImage(int& size)
{
  size = mySize;
  return &myImage[0];
}
