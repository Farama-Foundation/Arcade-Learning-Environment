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
// $Id: CartE7.hxx,v 1.10 2007/01/14 16:17:53 stephena Exp $
//============================================================================

#ifndef CARTRIDGEE7_HXX
#define CARTRIDGEE7_HXX

class System;
class Serializer;
class Deserializer;

#include "m6502/src/bspf/src/bspf.hxx"
#include "Cart.hxx"

/**
  This is the cartridge class for M-Network bankswitched games.  
  In this bankswitching scheme the 2600's 4K cartridge address 
  space is broken into two 2K segments.

  Kevin Horton describes E7 as follows:

    Only M-Network used this scheme. This has to be the 
    most complex method used in any cart! :-)  It allows 
    for the capability of 2K of RAM; although it doesn't 
    have to be used (in fact, only one cart used it).  
    There are now 8 2K banks, instead of 4.  The last 2K 
    in the cart always points to the last 2K of the ROM 
    image, while the first 2K is selectable.  You access 
    1FE0 to 1FE6 to select which 2K bank. Note that you
    cannot select the last 2K of the ROM image into the 
    lower 2K of the cart!  Accessing 1FE7 selects 1K of 
    RAM at 1000-17FF instead of ROM!  The 2K of RAM is
    broken up into two 1K sections.  One 1K section is 
    mapped in at 1000-17FF if 1FE7 has been accessed.  
    1000-13FF is the write port, while 1400-17FF is the 
    read port.  The second 1K of RAM appears at 1800-19FF.  
    1800-18FF is the write port while 1900-19FF is the 
    read port.  You select which 256 byte block appears 
    here by accessing 1FF8 to 1FFB.

  @author  Bradford W. Mott
  @version $Id: CartE7.hxx,v 1.10 2007/01/14 16:17:53 stephena Exp $
*/
class CartridgeE7 : public Cartridge
{
  public:
    /**
      Create a new cartridge using the specified image

      @param image Pointer to the ROM image
    */
    CartridgeE7(const uInt8* image);
 
    /**
      Destructor
    */
    virtual ~CartridgeE7();

  public:
    /**
      Get a null terminated string which is the device's name (i.e. "M6532")

      @return The name of the device
    */
    virtual const char* name() const;

    /**
      Reset device to its power-on state
    */
    virtual void reset();

    /**
      Install cartridge in the specified system.  Invoked by the system
      when the cartridge is attached to it.

      @param system The system the device should install itself in
    */
    virtual void install(System& system);

    /**
      Saves the current state of this device to the given Serializer.

      @param out The serializer device to save to.
      @return The result of the save.  True on success, false on failure.
    */
    virtual bool save(Serializer& out);

    /**
      Loads the current state of this device from the given Deserializer.

      @param in The deserializer device to load from.
      @return The result of the load.  True on success, false on failure.
    */
    virtual bool load(Deserializer& in);

    /**
      Install pages for the specified bank in the system.

      @param bank The bank that should be installed in the system
    */
    virtual void bank(uInt16 bank);

    /**
      Get the current bank.

      @return  The current bank, or -1 if bankswitching not supported
    */
    virtual int bank();

    /**
      Query the number of banks supported by the cartridge.
    */
    virtual int bankCount();

    /**
      Patch the cartridge ROM.

      @param address  The ROM address to patch
      @param value    The value to place into the address
      @return    Success or failure of the patch operation
    */
    virtual bool patch(uInt16 address, uInt8 value);

    /**
      Access the internal ROM image for this cartridge.

      @param size  Set to the size of the internal ROM image data
      @return  A pointer to the internal ROM image data
    */
    virtual uInt8* getImage(int& size);

  public:
    /**
      Get the byte at the specified address.

      @return The byte at the specified address
    */
    virtual uInt8 peek(uInt16 address);

    /**
      Change the byte at the specified address to the given value

      @param address The address where the value should be stored
      @param value The value to be stored at the address
    */
    virtual void poke(uInt16 address, uInt8 value);

  private:
    /**
      Install pages for the specified 256 byte bank of RAM

      @param bank The bank that should be installed in the system
    */
    void bankRAM(uInt16 bank);

  private:
    // Indicates which slice is in the segment
    uInt16 myCurrentSlice[2];

    // Indicates which 256 byte bank of RAM is being used
    uInt16 myCurrentRAM;

    // The 16K ROM image of the cartridge
    uInt8 myImage[16384];

    // The 2048 bytes of RAM
    uInt8 myRAM[2048];
};

#endif
