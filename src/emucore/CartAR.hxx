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
// $Id: CartAR.hxx,v 1.12 2007/01/14 16:17:53 stephena Exp $
//============================================================================

#ifndef CARTRIDGEAR_HXX
#define CARTRIDGEAR_HXX

class M6502High;
class System;
class Serializer;
class Deserializer;

#include "m6502/src/bspf/src/bspf.hxx"
#include "Cart.hxx"

/**
  This is the cartridge class for Arcadia (aka Starpath) Supercharger 
  games.  Christopher Salomon provided most of the technical details 
  used in creating this class.  A good description of the Supercharger
  is provided in the Cuttle Cart's manual.

  The Supercharger has four 2K banks.  There are three banks of RAM 
  and one bank of ROM.  All 6K of the RAM can be read and written.

  @author  Bradford W. Mott
  @version $Id: CartAR.hxx,v 1.12 2007/01/14 16:17:53 stephena Exp $
*/
class CartridgeAR : public Cartridge
{
  public:
    /**
      Create a new cartridge using the specified image and size

      @param image     Pointer to the ROM image
      @param size      The size of the ROM image
      @param fastbios  Whether or not to quickly execute the BIOS code
    */
    CartridgeAR(const uInt8* image, uInt32 size, bool fastbios);

    /**
      Destructor
    */
    virtual ~CartridgeAR();

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
      Notification method invoked by the system right before the
      system resets its cycle counter to zero.  It may be necessary
      to override this method for devices that remember cycle counts.
    */
    virtual void systemCyclesReset();

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
      Get the byte at the specified address

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
    // Handle a change to the bank configuration
    void bankConfiguration(uInt8 configuration);

    // Compute the sum of the array of bytes
    uInt8 checksum(uInt8* s, uInt16 length);

    // Load the specified load into SC RAM
    void loadIntoRAM(uInt8 load);

    // Sets up a "dummy" BIOS ROM in the ROM bank of the cartridge
    void initializeROM(bool fastbios);

  private:
    // Pointer to the 6502 processor in the system
    M6502High* my6502;

    // Indicates the offest within the image for the corresponding bank
    uInt32 myImageOffset[2];

    // The 6K of RAM and 2K of ROM contained in the Supercharger
    uInt8 myImage[8192];

    // The 256 byte header for the current 8448 byte load
    uInt8 myHeader[256];

    // All of the 8448 byte loads associated with the game 
    uInt8* myLoadImages;

    // Indicates how many 8448 loads there are
    uInt8 myNumberOfLoadImages;

    // Indicates if the RAM is write enabled
    bool myWriteEnabled;

    // Indicates if the ROM's power is on or off
    bool myPower;

    // Indicates when the power was last turned on
    Int32 myPowerRomCycle;

    // Data hold register used for writing
    uInt8 myDataHoldRegister;

    // Indicates number of distinct accesses when data hold register was set
    uInt32 myNumberOfDistinctAccesses;

    // Indicates if a write is pending or not
    bool myWritePending;

    uInt16 myCurrentBank;
};

#endif
