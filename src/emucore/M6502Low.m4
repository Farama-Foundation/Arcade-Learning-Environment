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
// Copyright (c) 1995-2005 by Bradford W. Mott and the Stella team
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: M6502Low.m4,v 1.4 2006/02/05 02:49:47 stephena Exp $
//============================================================================

/**
  Code to handle addressing modes and branch instructions for
  low compatibility emulation

  @author  Bradford W. Mott
  @version $Id: M6502Low.m4,v 1.4 2006/02/05 02:49:47 stephena Exp $
*/

#ifndef NOTSAMEPAGE
  #define NOTSAMEPAGE(_addr1, _addr2) (((_addr1) ^ (_addr2)) & 0xff00)
#endif

define(M6502_IMPLIED, `{
}')

define(M6502_IMMEDIATE_READ, `{
  operandAddress = PC++;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTE_READ, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTE_WRITE, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;
}')

define(M6502_ABSOLUTE_READMODIFYWRITE, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTEX_READ, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;

  // See if we need to add one cycle for indexing across a page boundary
  if(NOTSAMEPAGE(operandAddress, operandAddress + X))
  {
    mySystem->incrementCycles(mySystemCyclesPerProcessorCycle);
  }

  operandAddress += X;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTEX_WRITE, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;
  operandAddress += X; 
}')

define(M6502_ABSOLUTEX_READMODIFYWRITE, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;
  operandAddress += X;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTEY_READ, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;

  // See if we need to add one cycle for indexing across a page boundary
  if(NOTSAMEPAGE(operandAddress, operandAddress + Y))
  {
    mySystem->incrementCycles(mySystemCyclesPerProcessorCycle);
  }

  operandAddress += Y;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTEY_WRITE, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;
  operandAddress += Y; 
}')

define(M6502_ABSOLUTEY_READMODIFYWRITE, `{
  operandAddress = (uInt16)peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;
  operandAddress += Y;
  operand = peek(operandAddress);
}')

define(M6502_ZERO_READ, `{
  operandAddress = peek(PC++);
  operand = peek(operandAddress);
}')

define(M6502_ZERO_WRITE, `{
  operandAddress = peek(PC++);
}')

define(M6502_ZERO_READMODIFYWRITE, `{
  operandAddress = peek(PC++);
  operand = peek(operandAddress);
}')

define(M6502_ZEROX_READ, `{
  operandAddress = (uInt8)(peek(PC++) + X);
  operand = peek(operandAddress); 
}')

define(M6502_ZEROX_WRITE, `{
  operandAddress = (uInt8)(peek(PC++) + X);
}')

define(M6502_ZEROX_READMODIFYWRITE, `{
  operandAddress = (uInt8)(peek(PC++) + X);
  operand = peek(operandAddress);
}')

define(M6502_ZEROY_READ, `{
  operandAddress = (uInt8)(peek(PC++) + Y);
  operand = peek(operandAddress); 
}')

define(M6502_ZEROY_WRITE, `{
  operandAddress = (uInt8)(peek(PC++) + Y);
}')

define(M6502_ZEROY_READMODIFYWRITE, `{
  operandAddress = (uInt8)(peek(PC++) + Y);
  operand = peek(operandAddress);
}')

define(M6502_INDIRECT, `{
  uInt16 addr = peek(PC) | ((uInt16)peek(PC + 1) << 8);
  PC += 2;

  // Simulate the error in the indirect addressing mode!
  uInt16 high = NOTSAMEPAGE(addr, addr + 1) ? (addr & 0xff00) : (addr + 1);

  operandAddress = peek(addr) | ((uInt16)peek(high) << 8);
}')

define(M6502_INDIRECTX_READ, `{
  uInt8 pointer = peek(PC++) + X;
  operandAddress = peek(pointer) | ((uInt16)peek(pointer + 1) << 8);
  operand = peek(operandAddress);
}')

define(M6502_INDIRECTX_WRITE, `{
  uInt8 pointer = peek(PC++) + X;
  operandAddress = peek(pointer) | ((uInt16)peek(pointer + 1) << 8);
}')

define(M6502_INDIRECTX_READMODIFYWRITE, `{
  uInt8 pointer = peek(PC++) + X;
  operandAddress = peek(pointer) | ((uInt16)peek(pointer + 1) << 8);
  operand = peek(operandAddress);
}')

define(M6502_INDIRECTY_READ, `{
  uInt8 pointer = peek(PC++);
  operandAddress = (uInt16)peek(pointer) | ((uInt16)peek(pointer + 1) << 8); 

  if(NOTSAMEPAGE(operandAddress, operandAddress + Y))
  {
    mySystem->incrementCycles(mySystemCyclesPerProcessorCycle);
  }

  operandAddress += Y;
  operand = peek(operandAddress);
}')

define(M6502_INDIRECTY_WRITE, `{
  uInt8 pointer = peek(PC++);
  operandAddress = (uInt16)peek(pointer) | ((uInt16)peek(pointer + 1) << 8); 
  operandAddress += Y;
}')

define(M6502_INDIRECTY_READMODIFYWRITE, `{
  uInt8 pointer = peek(PC++);
  operandAddress = (uInt16)peek(pointer) | ((uInt16)peek(pointer + 1) << 8); 
  operandAddress += Y;
  operand = peek(operandAddress);
}')


define(M6502_BCC, `{
  if(!C)
  {
    uInt16 address = PC + (Int8)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BCS, `{
  if(C)
  {
    uInt16 address = PC + (Int8)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BEQ, `{
  if(!notZ)
  {
    uInt16 address = PC + (Int8)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BMI, `{
  if(N)
  {
    uInt16 address = PC + (Int8)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BNE, `{
  if(notZ)
  {
    uInt16 address = PC + (Int8)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BPL, `{
  if(!N)
  {
    uInt16 address = PC + (Int8)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BVC, `{
  if(!V)
  {
    uInt16 address = PC + (Int8)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BVS, `{
  if(V)
  {
    uInt16 address = PC + (Int8)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')


