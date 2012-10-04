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
// $Id: CartE0.hxx,v 1.9 2007/01/14 16:17:53 stephena Exp $
//============================================================================

#ifndef CARTRIDGEE0_HXX
#define CARTRIDGEE0_HXX

class System;
class Serializer;
class Deserializer;

#include "m6502/src/bspf/src/bspf.hxx"
#include "Cart.hxx"

/**
  This is the cartridge class for Parker Brothers' 8K games.  In 
  this bankswitching scheme the 2600's 4K cartridge address space 
  is broken into four 1K segments.  The desired 1K slice of the
  ROM is selected by accessing 1FE0 to 1FE7 for the first 1K.
  1FE8 to 1FEF selects the slice for the second 1K, and 1FF0 to 
  1FF8 selects the slice for the third 1K.   The last 1K segment 
  always points to the last 1K of the ROM image.
  
  @author  Bradford W. Mott
  @version $Id: CartE0.hxx,v 1.9 2007/01/14 16:17:53 stephena Exp $
*/
class CartridgeE0 : public Cartridge
{
  public:
    /**
      Create a new cartridge using the specified image

      @param image Pointer to the ROM image
    */
    CartridgeE0(const uInt8* image);
 
    /**
      Destructor
    */
    virtual ~CartridgeE0();

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
      Install the specified slice for segment zero

      @param slice The slice to map into the segment
    */
    void segmentZero(uInt16 slice);

    /**
      Install the specified slice for segment one

      @param slice The slice to map into the segment
    */
    void segmentOne(uInt16 slice);

    /**
      Install the specified slice for segment two

      @param slice The slice to map into the segment
    */
    void segmentTwo(uInt16 slice);

  private:
    // Indicates the slice mapped into each of the four segments
    uInt16 myCurrentSlice[4];

    // The 8K ROM image of the cartridge
    uInt8 myImage[8192];
};

#endif
