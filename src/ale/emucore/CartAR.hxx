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

namespace ale {
namespace stella {

class M6502High;
class System;
class Serializer;
class Deserializer;

}  // namespace stella
}  // namespace ale

#include "ale/emucore/Cart.hxx"

namespace ale {
namespace stella {

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
    CartridgeAR(const uint8_t* image, uint32_t size, bool fastbios);

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
    virtual void bank(uint16_t bank);

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
    virtual bool patch(uint16_t address, uint8_t value);

    /**
      Access the internal ROM image for this cartridge.

      @param size  Set to the size of the internal ROM image data
      @return  A pointer to the internal ROM image data
    */
    virtual uint8_t* getImage(int& size);

  public:
    /**
      Get the byte at the specified address

      @return The byte at the specified address
    */
    virtual uint8_t peek(uint16_t address);

    /**
      Change the byte at the specified address to the given value

      @param address The address where the value should be stored
      @param value The value to be stored at the address
    */
    virtual void poke(uint16_t address, uint8_t value);

  private:
    // Handle a change to the bank configuration
    void bankConfiguration(uint8_t configuration);

    // Compute the sum of the array of bytes
    uint8_t checksum(uint8_t* s, uint16_t length);

    // Load the specified load into SC RAM
    void loadIntoRAM(uint8_t load);

    // Sets up a "dummy" BIOS ROM in the ROM bank of the cartridge
    void initializeROM(bool fastbios);

  private:
    // Pointer to the 6502 processor in the system
    M6502High* my6502;

    // Indicates the offest within the image for the corresponding bank
    uint32_t myImageOffset[2];

    // The 6K of RAM and 2K of ROM contained in the Supercharger
    uint8_t myImage[8192];

    // The 256 byte header for the current 8448 byte load
    uint8_t myHeader[256];

    // All of the 8448 byte loads associated with the game
    uint8_t* myLoadImages;

    // Indicates how many 8448 loads there are
    uint8_t myNumberOfLoadImages;

    // Indicates if the RAM is write enabled
    bool myWriteEnabled;

    // Indicates if the ROM's power is on or off
    bool myPower;

    // Indicates when the power was last turned on
    int myPowerRomCycle;

    // Data hold register used for writing
    uint8_t myDataHoldRegister;

    // Indicates number of distinct accesses when data hold register was set
    uint32_t myNumberOfDistinctAccesses;

    // Indicates if a write is pending or not
    bool myWritePending;

    uint16_t myCurrentBank;
};

}  // namespace stella
}  // namespace ale

#endif
