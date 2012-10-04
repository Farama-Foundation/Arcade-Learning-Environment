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

class Console;
class System;
class Serializer;
class Deserializer;

#include "m6502/src/bspf/src/bspf.hxx"
#include "m6502/src/Device.hxx"
#include "Random.hxx"

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
    virtual uInt8 peek(uInt16 address);

    /**
      Change the byte at the specified address to the given value

      @param address The address where the value should be stored
      @param value The value to be stored at the address
    */
    virtual void poke(uInt16 address, uInt8 value);

  private:
    // Reference to the console
    const Console& myConsole;

    // An amazing 128 bytes of RAM
    uInt8 myRAM[128];

    // Current value of my Timer
    uInt32 myTimer;

    // Log base 2 of the number of cycles in a timer interval
    uInt32 myIntervalShift;

    // Indicates the number of cycles when the timer was last set
    Int32 myCyclesWhenTimerSet;

    // Indicates when the timer was read after timer interrupt occured
    Int32 myCyclesWhenInterruptReset;

    // Indicates if a read from timer has taken place after interrupt occured
    bool myTimerReadAfterInterrupt;

    // Data Direction Register for Port A
    uInt8 myDDRA;

    // Data Direction Register for Port B
    uInt8 myDDRB;

  private:
    // Copy constructor isn't supported by this class so make it private
    M6532(const M6532&);
 
    // Assignment operator isn't supported by this class so make it private
    M6532& operator = (const M6532&);
};
#endif

