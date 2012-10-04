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
// $Id: System.hxx,v 1.16 2007/01/01 18:04:51 stephena Exp $
//============================================================================

#ifndef SYSTEM_HXX
#define SYSTEM_HXX

class Device;
class M6502;
class TIA;
class NullDevice;
class Serializer;
class Deserializer;

#include "bspf/src/bspf.hxx"
#include "Device.hxx"
#include "NullDev.hxx"

/**
  This class represents a system consisting of a 6502 microprocessor
  and a set of devices.  The devices are mapped into an addressing
  space of 2^n bytes (1 <= n <= 16).  The addressing space is broken
  into 2^m byte pages (1 <= m <= n), where a page is the smallest unit
  a device can use when installing itself in the system.

  In general the addressing space will be 8192 (2^13) bytes for a 
  6507 based system and 65536 (2^16) bytes for a 6502 based system.

  TODO: To allow for dynamic code generation we probably need to
        add a tag to each page that indicates if it is read only
        memory.  We also need to notify the processor anytime a
        page access method is changed so that it can clear the
        dynamic code for that page of memory.

  @author  Bradford W. Mott
  @version $Id: System.hxx,v 1.16 2007/01/01 18:04:51 stephena Exp $
*/
class System
{
  public:
    /**
      Create a new system with an addressing space of 2^n bytes and
      pages of 2^m bytes.

      @param n Log base 2 of the addressing space size
      @param m Log base 2 of the page size
    */
    System(uInt16 n, uInt16 m);

    /**
      Destructor
    */
    virtual ~System();

  public:
    /**
      Reset the system cycle counter, the attached devices, and the
      attached processor of the system.
    */
    void reset();

    /**
      Saves the current state of this system class to the given Serializer.

      @param out The serializer device to save to.
      @return The result of the save.  True on success, false on failure.
    */
    bool save(Serializer& out);

    /**
      Loads the current state of this system class from the given Deserializer.

      @param in The deserializer device to load from.
      @return The result of the load.  True on success, false on failure.
    */
    bool load(Deserializer& in);

  public:
    /**
      Attach the specified device and claim ownership of it.  The device 
      will be asked to install itself.

      @param device The device to attach to the system
    */
    void attach(Device* device);

    /**
      Attach the specified processor and claim ownership of it.  The
      processor will be asked to install itself.

      @param m6502 The 6502 microprocessor to attach to the system
    */
    void attach(M6502* m6502);

    /**
      Attach the specified TIA device and claim ownership of it.  The device 
      will be asked to install itself.

      @param tia The TIA device to attach to the system
    */
    void attach(TIA* tia);

    /**
      Saves the current state of Stella to the given file.  Calls
      save on every device and CPU attached to this system.

      @param md5sum   MD5 of the current ROM
      @param out      The serializer device to save to

      @return  False on any errors, else true
    */
    bool saveState(const string& md5sum, Serializer& out);

    /**
      Loads the current state of Stella from the given file.  Calls
      load on every device and CPU attached to this system.

      @param md5sum   MD5 of the current ROM
      @param in       The deserializer device to load from

      @return  False on any errors, else true
    */
    bool loadState(const string& md5sum, Deserializer& in);

  public:
    /**
      Answer the 6502 microprocessor attached to the system.  If a
      processor has not been attached calling this function will fail.

      @return The attached 6502 microprocessor
    */
    M6502& m6502()
    {
      return *myM6502;
    }

    /**
      Answer the TIA device attached to the system.

      @return The attached TIA device
    */
    TIA& tia()
    {
      return *myTIA;
    }

    /**
      Get the null device associated with the system.  Every system 
      has a null device associated with it that's used by pages which 
      aren't mapped to "real" devices.

      @return The null device associated with the system
    */
    NullDevice& nullDevice()
    {
      return myNullDevice;
    }

    /**
      Get the total number of pages available in the system.

      @return The total number of pages available
    */
    uInt16 numberOfPages() const
    {
      return myNumberOfPages;
    }

    /**
      Get the amount to right shift an address by to obtain its page.

      @return The amount to right shift an address by to get its page
    */
    uInt16 pageShift() const
    {
      return myPageShift;
    }

    /**
      Get the mask to apply to an address to obtain its page offset.

      @return The mask to apply to an address to obtain its page offset
    */
    uInt16 pageMask() const
    {
      return myPageMask;
    }
 
  public:
    /**
      Get the number of system cycles which have passed since the last
      time cycles were reset or the system was reset.

      @return The number of system cycles which have passed
    */
    uInt32 cycles() const 
    { 
      return myCycles; 
    }

    /**
      Increment the system cycles by the specified number of cycles.

      @param amount The amount to add to the system cycles counter
    */
    void incrementCycles(uInt32 amount) 
    { 
      myCycles += amount; 
    }

    /**
      Reset the system cycle count to zero.  The first thing that
      happens is that all devices are notified of the reset by invoking 
      their systemCyclesReset method then the system cycle count is 
      reset to zero.
    */
    void resetCycles();

  public:
    /**
      Get the current state of the data bus in the system.  The current
      state is the last data that was accessed by the system.

      @return the data bus state
    */  
    uInt8 getDataBusState() const;

    /**
      Get the byte at the specified address.  No masking of the
      address occurs before it's sent to the device mapped at
      the address.

      @return The byte at the specified address
    */
    uInt8 peek(uInt16 address);

    /**
      Change the byte at the specified address to the given value.
      No masking of the address occurs before it's sent to the device
      mapped at the address.

      @param address The address where the value should be stored
      @param value The value to be stored at the address
    */
    void poke(uInt16 address, uInt8 value);

    /**
      Lock/unlock the data bus. When the bus is locked, peek() and
      poke() don't update the bus state. The bus should be unlocked
      while the CPU is running (normal emulation, or when the debugger
      is stepping/advancing). It should be locked while the debugger
      is active but not running the CPU. This is so the debugger can
      use System.peek() to examine memory/registers without changing
      the state of the system.
    */
    void lockDataBus();
    void unlockDataBus();

  public:
    /**
      Structure used to specify access methods for a page
    */
    struct PageAccess
    {
      /**
        Pointer to a block of memory or the null pointer.  The null pointer
        indicates that the device's peek method should be invoked for reads
        to this page, while other values are the base address of an array 
        to directly access for reads to this page.
      */
      uInt8* directPeekBase;

      /**
        Pointer to a block of memory or the null pointer.  The null pointer
        indicates that the device's poke method should be invoked for writes
        to this page, while other values are the base address of an array 
        to directly access for pokes to this page.
      */
      uInt8* directPokeBase;

      /**
        Pointer to the device associated with this page or to the system's 
        null device if the page hasn't been mapped to a device
      */
      Device* device;
    };

    /**
      Set the page accessing method for the specified page.

      @param page The page accessing methods should be set for
      @param access The accessing methods to be used by the page
    */
    void setPageAccess(uInt16 page, const PageAccess& access);

    /**
      Get the page accessing method for the specified page.

      @param page The page to get accessing methods for
      @return The accessing methods used by the page
    */
    const PageAccess& getPageAccess(uInt16 page);
 
  private:
    // Mask to apply to an address before accessing memory
    const uInt16 myAddressMask;

    // Amount to shift an address by to determine what page it's on
    const uInt16 myPageShift;

    // Mask to apply to an address to obtain its page offset
    const uInt16 myPageMask;
 
    // Number of pages in the system
    const uInt16 myNumberOfPages;

    // Pointer to a dynamically allocated array of PageAccess structures
    PageAccess* myPageAccessTable;

    // Array of all the devices attached to the system
    Device* myDevices[100];

    // Number of devices attached to the system
    uInt32 myNumberOfDevices;

    // 6502 processor attached to the system or the null pointer
    M6502* myM6502;

    // TIA device attached to the system or the null pointer
    TIA* myTIA;

    // Number of system cycles executed since the last reset
    uInt32 myCycles;

    // Null device to use for page which are not installed
    NullDevice myNullDevice; 

    // The current state of the Data Bus
    uInt8 myDataBusState;

    // Whether or not peek() updates the data bus state. This
    // is true during normal emulation, and false when the
    // debugger is active.
    bool myDataBusLocked;

  private:
    // Copy constructor isn't supported by this class so make it private
    System(const System&);

    // Assignment operator isn't supported by this class so make it private
    System& operator = (const System&);
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
inline uInt8 System::getDataBusState() const
{
  return myDataBusState;
}

#endif
