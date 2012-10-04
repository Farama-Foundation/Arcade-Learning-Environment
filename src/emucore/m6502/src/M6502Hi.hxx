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
// $Id: M6502Hi.hxx,v 1.5 2007/01/01 18:04:51 stephena Exp $
//============================================================================

#ifndef M6502HIGH_HXX
#define M6502HIGH_HXX

class M6502High;
class Serializer;
class Deserializer;

#include "bspf/src/bspf.hxx"
#include "M6502.hxx"

/**
  This class provides a high compatibility 6502 microprocessor emulator.  
  The memory accesses and cycle counts it generates are valid at the
  sub-instruction level and "false" reads are generated (such as the ones 
  produced by the Indirect,X addressing when it crosses a page boundary).
  This provides provides better compatibility for hardware that has side
  effects and for games which are very time sensitive.

  @author  Bradford W. Mott
  @version $Id: M6502Hi.hxx,v 1.5 2007/01/01 18:04:51 stephena Exp $
*/
class M6502High : public M6502
{
  public:
    /**
      Create a new high compatibility 6502 microprocessor with the 
      specified cycle multiplier.

      @param systemCyclesPerProcessorCycle The cycle multiplier
    */
    M6502High(uInt32 systemCyclesPerProcessorCycle);

    /**
      Destructor
    */
    virtual ~M6502High();

  public:
    /**
      Execute instructions until the specified number of instructions
      is executed, someone stops execution, or an error occurs.  Answers
      true iff execution stops normally.

      @param number Indicates the number of instructions to execute
      @return true iff execution stops normally
    */
    virtual bool execute(uInt32 number);

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
      Get a null terminated string which is the processors's name (i.e. "M6532")

      @return The name of the device
    */
    virtual const char* name() const;

  public:
    /**
      Get the number of memory accesses to distinct memory locations

      @return The number of memory accesses to distinct memory locations
    */
    uInt32 distinctAccesses() const
    {
      return myNumberOfDistinctAccesses;
    }

  protected:
    /**
      Called after an interrupt has be requested using irq() or nmi()
    */
    void interruptHandler();

  protected:
    /*
      Get the byte at the specified address and update the cycle
      count

      @return The byte at the specified address
    */
    inline uInt8 peek(uInt16 address);

    /**
      Change the byte at the specified address to the given value and
      update the cycle count

      @param address The address where the value should be stored
      @param value The value to be stored at the address
    */
    inline void poke(uInt16 address, uInt8 value);

  private:
    // Indicates the numer of distinct memory accesses
    uInt32 myNumberOfDistinctAccesses;

    // Indicates the last address which was accessed
    uInt16 myLastAddress;
};
#endif

