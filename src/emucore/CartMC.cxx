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

#include "Random.hxx"
#include "System.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"
#include "CartMC.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CartridgeMC::CartridgeMC(const uInt8* image, uInt32 size)
  : mySlot3Locked(false)
{
  uInt32 i;

  // Make sure size is reasonable
  assert(size <= 128 * 1024);

  // Allocate array for the cart's RAM
  myRAM = new uInt8[32 * 1024];

  // Initialize RAM with random values
  class Random random;
  for(i = 0; i < 32 * 1024; ++i)
  {
    myRAM[i] = random.next();
  }

  // Allocate array for the ROM image
  myImage = new uInt8[128 * 1024];

  // Set the contents of the entire ROM to 0
  for(i = 0; i < 128 * 1024; ++i)
  {
    myImage[i] = 0;
  }

  // Copy the ROM image to the end of the ROM buffer
  for(i = 0; i < size; ++i)
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
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeMC::install(System& system)
{
  mySystem = &system;
  uInt16 shift = mySystem->pageShift();
  uInt16 mask = mySystem->pageMask();

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
  for(uInt32 i = 0x00; i < 0x40; i += (1 << shift))
  {
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    access.device = this;
    mySystem->setPageAccess(i >> shift, access);
  }

  // Map the cartridge into the system
  for(uInt32 j = 0x1000; j < 0x2000; j += (1 << shift))
  {
    access.device = this;
    access.directPeekBase = 0;
    access.directPokeBase = 0;
    mySystem->setPageAccess(j >> shift, access);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8 CartridgeMC::peek(uInt16 address)
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
    uInt8 block;

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
      return myImage[(uInt32)(block & 0x7F) * 1024 + (address & 0x03FF)];
    }
    else
    {
      // This is a RAM access, however, is it to the read or write port?
      if(address & 0x0200)
      {
        // Reading from the read port of the RAM block
        return myRAM[(uInt32)(block & 0x3F) * 512 + (address & 0x01FF)];
      }
      else
      {
        // Oops, reading from the write port of the RAM block!
        myRAM[(uInt32)(block & 0x3F) * 512 + (address & 0x01FF)] = 0;
        return 0;
      }
    }
  }  
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeMC::poke(uInt16 address, uInt8 value)
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
    uInt8 block;

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
      myRAM[(uInt32)(block & 0x3F) * 512 + (address & 0x01FF)] = value;
    }
  }  
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool CartridgeMC::save(Serializer& out)
{
  uInt32 i;
  string cart = name();

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
bool CartridgeMC::load(Deserializer& in)
{
  uInt32 i;
  string cart = name();

  try
  {
    uInt32 limit;

    if(in.getString() != cart)
      return false;

    // The currentBlock array
    limit = (uInt32) in.getInt();
    for(i = 0; i < limit; ++i)
      myCurrentBlock[i] = (uInt8) in.getInt();

    // The 32K of RAM
    limit = (uInt32) in.getInt();
    for(i = 0; i < limit; ++i)
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

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CartridgeMC::bank(uInt16 b)
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
bool CartridgeMC::patch(uInt16 address, uInt8 value)
{
  // TODO: implement
  return false;
} 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt8* CartridgeMC::getImage(int& size)
{
  size = 128 * 1024; // FIXME: keep track of original size
  return &myImage[0];
}
