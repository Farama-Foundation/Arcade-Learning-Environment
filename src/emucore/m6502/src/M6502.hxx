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
// $Id: M6502.hxx,v 1.20 2007/01/01 18:04:51 stephena Exp $
//============================================================================

#ifndef M6502_HXX
#define M6502_HXX

class D6502;
class M6502;
class Serializer;
class Deserializer;
class Debugger;
class CpuDebug;
class Expression;
class PackedBitArray;

#include "bspf/src/bspf.hxx"
#include "System.hxx"
#include "Array.hxx"
#ifdef DEBUGGER_SUPPORT
#include "StringList.hxx"
#endif

typedef Common::Array<Expression*> ExpressionList;

/**
  This is an abstract base class for classes that emulate the
  6502 microprocessor.  The 6502 is an 8-bit microprocessor that
  has a 64K addressing space.

  @author  Bradford W. Mott
  @version $Id: M6502.hxx,v 1.20 2007/01/01 18:04:51 stephena Exp $ 
*/
class M6502
{
  public:
    /**
      The 6502 debugger class is a friend who needs special access
    */
    friend class CpuDebug;

  public:
    /**
      Enumeration of the 6502 addressing modes
    */
    enum AddressingMode 
    {
      Absolute, AbsoluteX, AbsoluteY, Immediate, Implied,
      Indirect, IndirectX, IndirectY, Invalid, Relative,
      Zero, ZeroX, ZeroY
    };

  public:
    /**
      Create a new 6502 microprocessor with the specified cycle 
      multiplier.  The cycle multiplier is the number of system cycles 
      per processor cycle.

      @param systemCyclesPerProcessorCycle The cycle multiplier
    */
    M6502(uInt32 systemCyclesPerProcessorCycle);

    /**
      Destructor
    */
    virtual ~M6502();

  public:
    /**
      Install the processor in the specified system.  Invoked by the
      system when the processor is attached to it.

      @param system The system the processor should install itself in
    */
    virtual void install(System& system);

    /**
      Reset the processor to its power-on state.  This method should not 
      be invoked until the entire 6502 system is constructed and installed
      since it involves reading the reset vector from memory.
    */
    virtual void reset();

    /**
      Request a maskable interrupt
    */
    virtual void irq();

    /**
      Request a non-maskable interrupt
    */
    virtual void nmi();

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

    /**
      Get a null terminated string which is the processor's name (i.e. "M6532")

      @return The name of the device
    */
    virtual const char* name() const = 0;

  public:
    /**
      Get the addressing mode of the specified instruction

      @param opcode The opcode of the instruction
      @return The addressing mode of the instruction
    */
    AddressingMode addressingMode(uInt8 opcode) const;

  public:
    /**
      Execute instructions until the specified number of instructions
      is executed, someone stops execution, or an error occurs.  Answers
      true iff execution stops normally.

      @param number Indicates the number of instructions to execute
      @return true iff execution stops normally
    */
    virtual bool execute(uInt32 number) = 0;

    /**
      Tell the processor to stop executing instructions.  Invoking this 
      method while the processor is executing instructions will stop 
      execution as soon as possible.
    */
    void stop();

    /**
      Answer true iff a fatal error has occured from which the processor
      cannot recover (i.e. illegal instruction, etc.)

      @return true iff a fatal error has occured
    */
    bool fatalError() const
    {
      return myExecutionStatus & FatalErrorBit;
    }
  
    /**
      Get the 16-bit value of the Program Counter register.

      @return The program counter register
    */
    uInt16 getPC() const { return PC; }

    /**
      Answer true iff the last memory access was a read.

      @return true iff last access was a read.
    */ 
    bool lastAccessWasRead() const { return myLastAccessWasRead; }

  public:
    /**
      Overload the ostream output operator for addressing modes.

      @param out The stream to output the addressing mode to
      @param mode The addressing mode to output
    */
    friend ostream& operator<<(ostream& out, const AddressingMode& mode);

  public:
#ifdef DEBUGGER_SUPPORT
    /**
      Attach the specified debugger.

      @param debugger The debugger to attach to the microprocessor.
    */
    void attach(Debugger& debugger);

    // TODO - document these methods
    void setBreakPoints(PackedBitArray *bp);
    void setTraps(PackedBitArray *read, PackedBitArray *write);
    int totalInstructionCount() { return myTotalInstructionCount; }

    unsigned int addCondBreak(Expression *e, string name);
    void delCondBreak(unsigned int brk);
    void clearCondBreaks();
    const StringList& getCondBreakNames();
    int evalCondBreaks();
#endif

  protected:
    /**
      Get the 8-bit value of the Processor Status register.

      @return The processor status register
    */
    uInt8 PS() const;

    /**
      Change the Processor Status register to correspond to the given value.

      @param ps The value to set the processor status register to
    */
    void PS(uInt8 ps);

  protected:
    uInt8 A;    // Accumulator
    uInt8 X;    // X index register
    uInt8 Y;    // Y index register
    uInt8 SP;   // Stack Pointer
    uInt8 IR;   // Instruction register
    uInt16 PC;  // Program Counter

    bool N;     // N flag for processor status register
    bool V;     // V flag for processor status register
    bool B;     // B flag for processor status register
    bool D;     // D flag for processor status register
    bool I;     // I flag for processor status register
    bool notZ;  // Z flag complement for processor status register
    bool C;     // C flag for processor status register

#ifdef DEBUGGER_SUPPORT
    /// Pointer to the debugger for this processor or the null pointer
    Debugger* myDebugger;

    PackedBitArray* myBreakPoints;
    PackedBitArray* myReadTraps;
    PackedBitArray* myWriteTraps;

    // Did we just now hit a trap?
    bool myJustHitTrapFlag;
    struct HitTrapInfo {
      string message;
      int address;
    };
    HitTrapInfo myHitTrapInfo;

    StringList myBreakCondNames;
    ExpressionList myBreakConds;
#endif

    /** 
      Bit fields used to indicate that certain conditions need to be 
      handled such as stopping execution, fatal errors, maskable interrupts 
      and non-maskable interrupts
    */
    uInt8 myExecutionStatus;

    /**
      Constants used for setting bits in myExecutionStatus
    */
    enum 
    {
      StopExecutionBit = 0x01,
      FatalErrorBit = 0x02,
      MaskableInterruptBit = 0x04,
      NonmaskableInterruptBit = 0x08
    };
  
    /// Pointer to the system the processor is installed in or the null pointer
    System* mySystem;

    /// Indicates the number of system cycles per processor cycle 
    const uInt32 mySystemCyclesPerProcessorCycle;

    /// Table of system cycles for each instruction
    uInt32 myInstructionSystemCycleTable[256]; 

    /// Indicates if the last memory access was a read or not
    bool myLastAccessWasRead;

  protected:
    /// Addressing mode for each of the 256 opcodes
    static AddressingMode ourAddressingModeTable[256];

    /// Lookup table used for binary-code-decimal math
    static uInt8 ourBCDTable[2][256];

    /**
      Table of instruction processor cycle times.  In some cases additional 
      cycles will be added during the execution of an instruction.
    */
    static uInt32 ourInstructionProcessorCycleTable[256];

    /// Table of instruction mnemonics
    static const char* ourInstructionMnemonicTable[256];

    int myTotalInstructionCount;
};

#endif
