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
// $Id: CartF8SC.cxx,v 1.13 2007/01/14 16:17:54 stephena Exp $
//============================================================================

#include <cassert>

#include "Random.hxx"
#include "System.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"
#include "CartF8SC.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeF8SC::CartridgeF8SC(const uInt8* image)
{
  // Copy the ROM image into my buffer
  for(uInt32 addr = 0; addr < 8192; ++addr)
  {
    myImage[addr] = image[addr];
  }

  // Initialize RAM with random values
  class Random random;
  for(uInt32 i = 0; i < 128; ++i)
  {
    myRAM[i] = random.next();
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeF8SC::~CartridgeF8SC()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* CartridgeF8SC::name() const
{
  return "CartridgeF8SC";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF8SC::reset()
{
  // Upon reset we switch to bank 1
  bank(1);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF8SC::install(System& system)
{
  mySystem = &system;
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert(((0x1080 & mask) == 0) && ((0x1100 & mask) == 0));

  // Set the page accessing methods for the hot spots
  System::PageAccess access;
  for(uInt32 i = (0x1FF8 & ~mask); i < 0x2000; i += (1 << shift))
  {
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    access.device = this;
    mySystem->setPageAccess(i >> shift, access);
  }

  // Set the page accessing method for the RAM writing pages
  for(uInt32 j = 0x1000; j < 0x1080; j += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = 0;
    access.directPokeBase = &myRAM[j & 0x007F];
    mySystem->setPageAccess(j >> shift, access);
  }
 
  // Set the page accessing method for the RAM reading pages
  for(uInt32 k = 0x1080; k < 0x1100; k += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = &myRAM[k & 0x007F];
    access.directPokeBase = 0;
    mySystem->setPageAccess(k >> shift, access);
  }

  // Install pages for bank 1
  bank(1);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8 CartridgeF8SC::peek(uInt16 address)
{
  address = address & 0x0FFF;

  if(!bankLocked) {
    // Switch banks if necessary
    switch(address)
    {
      case 0x0FF8:
        // Set the current bank to the lower 4k bank
        bank(0);
        break;
  
      case 0x0FF9:
        // Set the current bank to the upper 4k bank
        bank(1);
        break;
  
      default:
        break;
    }
  }

  // NOTE: This does not handle accessing RAM, however, this function
  // should never be called for RAM because of the way page accessing
  // has been setup
  return myImage[myCurrentBank * 4096 + address];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF8SC::poke(uInt16 address, uInt8)
{
  address = address & 0x0FFF;

  if(!bankLocked) {
    // Switch banks if necessary
    switch(address)
    {
      case 0x0FF8:
        // Set the current bank to the lower 4k bank
        bank(0);
        break;
  
      case 0x0FF9:
        // Set the current bank to the upper 4k bank
        bank(1);
        break;
  
      default:
        break;
    }
  }

  // NOTE: This does not handle accessing RAM, however, this function
  // should never be called for RAM because of the way page accessing
  // has been setup
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeF8SC::save(Serializer& out)
{
  string cart = name();

  try
  {
    out.putString(cart);

    out.putInt(myCurrentBank);

    // The 128 bytes of RAM
    out.putInt(128);
    for(uInt32 i = 0; i < 128; ++i)
      out.putInt(myRAM[i]);
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
bool CartridgeF8SC::load(Deserializer& in)
{
  string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    myCurrentBank = (uInt16) in.getInt();

    uInt32 limit = (uInt32) in.getInt();
    for(uInt32 i = 0; i < limit; ++i)
      myRAM[i] = (uInt8) in.getInt();
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

  // Remember what bank we were in
  bank(myCurrentBank);

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF8SC::bank(uInt16 bank)
{ 
  if(bankLocked) return;

  // Remember what bank we're in
  myCurrentBank = bank;
  uInt16 offset = myCurrentBank << 12;
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  // Map ROM image into the system
  for(uInt32 address = 0x1100; address < (0x1FF8U & ~mask);
      address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x0FFF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeF8SC::bank()
{
  return myCurrentBank;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeF8SC::bankCount()
{
  return 2;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeF8SC::patch(uInt16 address, uInt8 value)
{
  address = address & 0x0FFF;
  myImage[myCurrentBank * 4096 + address] = value;
  return true;
} 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8* CartridgeF8SC::getImage(int& size)
{
  size = 8192;
  return &myImage[0];
}
