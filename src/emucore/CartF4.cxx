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
// $Id: CartF4.cxx,v 1.11 2007/01/14 16:17:53 stephena Exp $
//============================================================================

#include <cassert>

#include "System.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"
#include "CartF4.hxx"
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeF4::CartridgeF4(const uInt8* image)
{
  // Copy the ROM image into my buffer
  for(uInt32 addr = 0; addr < 32768; ++addr)
  {
    myImage[addr] = image[addr];
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeF4::~CartridgeF4()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* CartridgeF4::name() const
{
  return "CartridgeF4";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF4::reset()
{
  // Upon reset we switch to bank 0
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF4::install(System& system)
{
  mySystem = &system;
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert((0x1000 & mask) == 0);

  // Set the page accessing methods for the hot spots
  System::PageAccess access;
  for(uInt32 i = (0x1FF4 & ~mask); i < 0x2000; i += (1 << shift))
  {
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    access.device = this;
    mySystem->setPageAccess(i >> shift, access);
  }

  // Install pages for bank 0
  bank(0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8 CartridgeF4::peek(uInt16 address)
{
  address = address & 0x0FFF;

  // Switch banks if necessary
  if((address >= 0x0FF4) && (address <= 0x0FFB))
  {
    bank(address - 0x0FF4);
  }

  return myImage[myCurrentBank * 4096 + address];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF4::poke(uInt16 address, uInt8)
{
  address = address & 0x0FFF;

  // Switch banks if necessary
  if((address >= 0x0FF4) && (address <= 0x0FFB))
  {
    bank(address - 0x0FF4);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeF4::save(Serializer& out)
{
  string cart = name();

  try
  {
    out.putString(cart);
    out.putInt(myCurrentBank);
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
bool CartridgeF4::load(Deserializer& in)
{
  string cart = name();

  try
  {
    if(in.getString() != cart)
    {
      return false;
    }

    myCurrentBank = (uInt16)in.getInt();
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

  // Remember what bank we were in
  bank(myCurrentBank);

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeF4::bank(uInt16 bank)
{ 
  if(bankLocked) return;

  // Remember what bank we're in
  myCurrentBank = bank;
  uInt16 offset = myCurrentBank * 4096;
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  // Map ROM image into the system
  for(uInt32 address = 0x1000; address < (0x1FF4U & ~mask);
      address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x0FFF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeF4::bank()
{
  return myCurrentBank;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int CartridgeF4::bankCount()
{
  return 8;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeF4::patch(uInt16 address, uInt8 value)
{
  address = address & 0x0FFF;
  myImage[myCurrentBank * 4096 + address] = value;
  return true;
} 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8* CartridgeF4::getImage(int& size)
{
  size = 32768;
  return &myImage[0];
}
