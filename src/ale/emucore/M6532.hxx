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
// $Id: M6532.hxx,v 1.5 2007/01/01 18:04:48 stephena Exp $
//============================================================================

#ifndef M6532_HXX
#define M6532_HXX

namespace ale {
namespace stella {

class Console;
class System;
class Serializer;
class Deserializer;

}  // namespace stella
}  // namespace ale

#include "ale/emucore/Device.hxx"
#include "ale/emucore/Random.hxx"

namespace ale {
namespace stella {

/**
  RIOT

  @author  Bradford W. Mott
  @version $Id: M6532.hxx,v 1.5 2007/01/01 18:04:48 stephena Exp $
*/
class M6532 : public Device
{
  public:
    /**
      Create a new 6532 for the specified console

      @param console The console the 6532 is associated with
    */
    M6532(const Console& console);

    /**
      Destructor
    */
    virtual ~M6532();

   public:
    /**
      Get a null terminated string which is the device's name (i.e. "M6532")

      @return The name of the device
    */
    virtual const char* name() const;

    /**
      Reset cartridge to its power-on state
    */
    virtual void reset();

    /**
      Notification method invoked by the system right before the
      system resets its cycle counter to zero.  It may be necessary
      to override this method for devices that remember cycle counts.
    */
    virtual void systemCyclesReset();

    /**
      Install 6532 in the specified system.  Invoked by the system
      when the 6532 is attached to it.

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
    // Reference to the console
    const Console& myConsole;

    // An amazing 128 bytes of RAM
    uint8_t myRAM[128];

    // Current value of my Timer
    uint32_t myTimer;

    // Log base 2 of the number of cycles in a timer interval
    uint32_t myIntervalShift;

    // Indicates the number of cycles when the timer was last set
    int myCyclesWhenTimerSet;

    // Indicates when the timer was read after timer interrupt occured
    int myCyclesWhenInterruptReset;

    // Indicates if a read from timer has taken place after interrupt occured
    bool myTimerReadAfterInterrupt;

    // Data Direction Register for Port A
    uint8_t myDDRA;

    // Data Direction Register for Port B
    uint8_t myDDRB;

  private:
    // Copy constructor isn't supported by this class so make it private
    M6532(const M6532&);

    // Assignment operator isn't supported by this class so make it private
    M6532& operator = (const M6532&);
};

}  // namespace stella
}  // namespace ale

#endif
