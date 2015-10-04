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

#include "System.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"
#include "CartE0.hxx"
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeE0::CartridgeE0(const uInt8* image)
{
  // Copy the ROM image into my buffer
  for(uInt32 addr = 0; addr < 8192; ++addr)
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
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert(((0x1000 & mask) == 0) && ((0x1400 & mask) == 0) &&
      ((0x1800 & mask) == 0) && ((0x1C00 & mask) == 0));

  // Set the page acessing methods for the first part of the last segment
  System::PageAccess access;
  access.directPokeBase = 0;
  access.device = this;
  for(uInt32 i = 0x1C00; i < (0x1FE0U & ~mask); i += (1 << shift))
  {
    access.directPeekBase = &myImage[7168 + (i & 0x03FF)];
    mySystem->setPageAccess(i >> shift, access);
  }
  myCurrentSlice[3] = 7;

  // Set the page accessing methods for the hot spots in the last segment
  access.directPeekBase = 0;
  access.directPokeBase = 0;
  access.device = this;
  for(uInt32 j = (0x1FE0 & ~mask); j < 0x2000; j += (1 << shift))
  {
    mySystem->setPageAccess(j >> shift, access);
  }

  // Install some default slices for the other segments
  segmentZero(4);
  segmentOne(5);
  segmentTwo(6);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8 CartridgeE0::peek(uInt16 address)
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
void CartridgeE0::poke(uInt16 address, uInt8)
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
void CartridgeE0::segmentZero(uInt16 slice)
{ 
  // Remember the new slice
  myCurrentSlice[0] = slice;
  uInt16 offset = slice << 10;
  uInt16 shift = mySystem->pageShift();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  for(uInt32 address = 0x1000; address < 0x1400; address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x03FF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE0::segmentOne(uInt16 slice)
{ 
  // Remember the new slice
  myCurrentSlice[1] = slice;
  uInt16 offset = slice << 10;
  uInt16 shift = mySystem->pageShift();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  for(uInt32 address = 0x1400; address < 0x1800; address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x03FF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeE0::segmentTwo(uInt16 slice)
{ 
  // Remember the new slice
  myCurrentSlice[2] = slice;
  uInt16 offset = slice << 10;
  uInt16 shift = mySystem->pageShift();

  // Setup the page access methods for the current bank
  System::PageAccess access;
  access.device = this;
  access.directPokeBase = 0;

  for(uInt32 address = 0x1800; address < 0x1C00; address += (1 << shift))
  {
    access.directPeekBase = &myImage[offset + (address & 0x03FF)];
    mySystem->setPageAccess(address >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeE0::save(Serializer& out)
{
  string cart = name();

  try
  {
    out.putString(cart);

    out.putInt(4);
    for(uInt32 i = 0; i < 4; ++i)
      out.putInt(myCurrentSlice[i]);
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
bool CartridgeE0::load(Deserializer& in)
{
  string cart = name();

  try
  {
    if(in.getString() != cart)
      return false;

    uInt32 limit = (uInt32) in.getInt();
    for(uInt32 i = 0; i < limit; ++i)
      myCurrentSlice[i] = (uInt16) in.getInt();
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
void CartridgeE0::bank(uInt16 bank)
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
bool CartridgeE0::patch(uInt16 address, uInt8 value)
{
  address = address & 0x0FFF;
  myImage[(myCurrentSlice[address >> 10] << 10) + (address & 0x03FF)] = value;
  return true;
} 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8* CartridgeE0::getImage(int& size)
{
  size = 8192;
  return &myImage[0];
}
