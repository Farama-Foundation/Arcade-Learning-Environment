//============================================================================
//
// MM     MM  6666  555555  0000   2222
// MMMM MMMM 66  66 55     00  00 22  22
// MM MMM MM 66     55     00  00     22
// MM  M  MM 66666  55555  00  00  22222  --  "A 6502 Microprocessor Emulator"
// MM     MM 66  66     55 00  00 22
// MM     MM 66  66 55  55 00  00 22
// MM     MM  6666   5555   0000  222222
//
// Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: Device.hxx,v 1.6 2007/01/14 16:17:57 stephena Exp $
//============================================================================

#ifndef DEVICE_HXX
#define DEVICE_HXX

class System;
class Serializer;
class Deserializer;

#include "bspf/src/bspf.hxx"

/**
  Abstract base class for devices which can be attached to a 6502
  based system.

  @author  Bradford W. Mott
  @version $Id: Device.hxx,v 1.6 2007/01/14 16:17:57 stephena Exp $
*/
class Device
{
  public:
    /**
      Create a new device
    */
    Device();

    /**
      Destructor
    */
    virtual ~Device();

  public:
    /**
      Get a null terminated string which is the device's name (i.e. "M6532")

      @return The name of the device
    */
    virtual const char* name() const = 0;

    /**
      Reset device to its power-on state
    */
    virtual void reset() = 0;

    /**
      Notification method invoked by the system right before the
      system resets its cycle counter to zero.  It may be necessary 
      to override this method for devices that remember cycle counts.
    */
    virtual void systemCyclesReset();

    /**
      Install device in the specified system.  Invoked by the system
      when the device is attached to it.

      @param system The system the device should install itself in
    */
    virtual void install(System& system) = 0;

    /**
      Saves the current state of this device to the given Serializer.

      @param out The serializer device to save to.
      @return The result of the save.  True on success, false on failure.
    */
    virtual bool save(Serializer& out) = 0;

    /**
      Loads the current state of this device from the given Deserializer.

      @param in The deserializer device to load from.
      @return The result of the load.  True on success, false on failure.
    */
    virtual bool load(Deserializer& in) = 0;

  public:
    /**
      Get the byte at the specified address

      @return The byte at the specified address
    */
    virtual uInt8 peek(uInt16 address) = 0;

    /**
      Change the byte at the specified address to the given value

      @param address The address where the value should be stored
      @param value The value to be stored at the address
    */
    virtual void poke(uInt16 address, uInt8 value) = 0;

  protected:
    /// Pointer to the system the device is installed in or the null pointer
    System* mySystem;
};

#endif
