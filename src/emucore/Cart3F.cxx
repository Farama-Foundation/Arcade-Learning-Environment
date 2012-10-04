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
// $Id: Cart3F.cxx,v 1.15 2007/01/14 16:17:52 stephena Exp $
//============================================================================

#include <cassert>

#include "System.hxx"
#include "TIA.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"
#include "Cart3F.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge3F::Cartridge3F(const uInt8* image, uInt32 size)
  : mySize(size)
{
  // Allocate array for the ROM image
  myImage = new uInt8[mySize];

  // Copy the ROM image into my buffer
  for(uInt32 addr = 0; addr < mySize; ++addr)
  {
    myImage[addr] = image[addr];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge3F::~Cartridge3F()
{
  delete[] myImage;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* Cartridge3F::name() const
{
  return "Cartridge3F";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3F::reset()
{
  // We'll map bank 0 into the first segment upon reset
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3F::install(System& system)
{
  mySystem = &system;
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert((0x1800 & mask) == 0);

  // Set the page accessing methods for the hot spots (for 100% emulation
  // we need to chain any accesses below 0x40 to the TIA. Our poke() method
  // does this via mySystem->tiaPoke(...), at least until we come up with a
  // cleaner way to do it.)
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
uInt8 Cartridge3F::peek(uInt16 address)
{
  address = address & 0x0FFF;

  if(address < 0x0800)
  {
    return myImage[(address & 0x07FF) + myCurrentBank * 2048];
  }
  else
  {
    return myImage[(address & 0x07FF) + mySize - 2048];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3F::poke(uInt16 address, uInt8 value)
{
  address = address & 0x0FFF;

  // Switch banks if necessary
  if(address <= 0x003F)
  {
    bank(value);
  }

  mySystem->tia().poke(address, value);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge3F::save(Serializer& out)
{
  string cart = name();

  try
  {
    out.putString(cart);
    out.putInt(myCurrentBank);
  }
  catch(const char* msg)
  {
    cerr << msg << endl;
    return false;
  }
  catch(...)
  {
    cerr << "Unknown error in save state for " << cart << endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge3F::load(Deserializer& in)
{
  string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    myCurrentBank = (uInt16) in.getInt();
  }
  catch(const char* msg)
  {
    cerr << msg << endl;
    return false;
  }
  catch(...)
  {
    cerr << "Unknown error in load state for " << cart << endl;
    return false;
  }

  // Now, go to the current bank
  bank(myCurrentBank);

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Cartridge3F::bank(uInt16 bank)
{ 
  if(bankLocked) return;

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

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Cartridge3F::bank()
{
  return myCurrentBank;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Cartridge3F::bankCount()
{
  return mySize / 2048;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge3F::patch(uInt16 address, uInt8 value)
{
  address = address & 0x0FFF;
  if(address < 0x0800)
  {
    myImage[(address & 0x07FF) + myCurrentBank * 2048] = value;
  }
  else
  {
    myImage[(address & 0x07FF) + mySize - 2048] = value;
  }
  return true;
} 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8* Cartridge3F::getImage(int& size)
{
  size = mySize;
  return &myImage[0];
}
