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
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
  PC += 2;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTE_WRITE, `{
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
  PC += 2;
}')

define(M6502_ABSOLUTE_READMODIFYWRITE, `{
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
  PC += 2;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTEX_READ, `{
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
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
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
  PC += 2;
  operandAddress += X;
}')

define(M6502_ABSOLUTEX_READMODIFYWRITE, `{
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
  PC += 2;
  operandAddress += X;
  operand = peek(operandAddress);
}')

define(M6502_ABSOLUTEY_READ, `{
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
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
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
  PC += 2;
  operandAddress += Y;
}')

define(M6502_ABSOLUTEY_READMODIFYWRITE, `{
  operandAddress = (uint16_t)peek(PC) | ((uint16_t)peek(PC + 1) << 8);
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
  operandAddress = (uint8_t)(peek(PC++) + X);
  operand = peek(operandAddress);
}')

define(M6502_ZEROX_WRITE, `{
  operandAddress = (uint8_t)(peek(PC++) + X);
}')

define(M6502_ZEROX_READMODIFYWRITE, `{
  operandAddress = (uint8_t)(peek(PC++) + X);
  operand = peek(operandAddress);
}')

define(M6502_ZEROY_READ, `{
  operandAddress = (uint8_t)(peek(PC++) + Y);
  operand = peek(operandAddress);
}')

define(M6502_ZEROY_WRITE, `{
  operandAddress = (uint8_t)(peek(PC++) + Y);
}')

define(M6502_ZEROY_READMODIFYWRITE, `{
  operandAddress = (uint8_t)(peek(PC++) + Y);
  operand = peek(operandAddress);
}')

define(M6502_INDIRECT, `{
  uint16_t addr = peek(PC) | ((uint16_t)peek(PC + 1) << 8);
  PC += 2;

  // Simulate the error in the indirect addressing mode!
  uint16_t high = NOTSAMEPAGE(addr, addr + 1) ? (addr & 0xff00) : (addr + 1);

  operandAddress = peek(addr) | ((uint16_t)peek(high) << 8);
}')

define(M6502_INDIRECTX_READ, `{
  uint8_t pointer = peek(PC++) + X;
  operandAddress = peek(pointer) | ((uint16_t)peek(pointer + 1) << 8);
  operand = peek(operandAddress);
}')

define(M6502_INDIRECTX_WRITE, `{
  uint8_t pointer = peek(PC++) + X;
  operandAddress = peek(pointer) | ((uint16_t)peek(pointer + 1) << 8);
}')

define(M6502_INDIRECTX_READMODIFYWRITE, `{
  uint8_t pointer = peek(PC++) + X;
  operandAddress = peek(pointer) | ((uint16_t)peek(pointer + 1) << 8);
  operand = peek(operandAddress);
}')

define(M6502_INDIRECTY_READ, `{
  uint8_t pointer = peek(PC++);
  operandAddress = (uint16_t)peek(pointer) | ((uint16_t)peek(pointer + 1) << 8);

  if(NOTSAMEPAGE(operandAddress, operandAddress + Y))
  {
    mySystem->incrementCycles(mySystemCyclesPerProcessorCycle);
  }

  operandAddress += Y;
  operand = peek(operandAddress);
}')

define(M6502_INDIRECTY_WRITE, `{
  uint8_t pointer = peek(PC++);
  operandAddress = (uint16_t)peek(pointer) | ((uint16_t)peek(pointer + 1) << 8);
  operandAddress += Y;
}')

define(M6502_INDIRECTY_READMODIFYWRITE, `{
  uint8_t pointer = peek(PC++);
  operandAddress = (uint16_t)peek(pointer) | ((uint16_t)peek(pointer + 1) << 8);
  operandAddress += Y;
  operand = peek(operandAddress);
}')


define(M6502_BCC, `{
  if(!C)
  {
    uint16_t address = PC + (int8_t)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BCS, `{
  if(C)
  {
    uint16_t address = PC + (int8_t)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BEQ, `{
  if(!notZ)
  {
    uint16_t address = PC + (int8_t)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BMI, `{
  if(N)
  {
    uint16_t address = PC + (int8_t)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BNE, `{
  if(notZ)
  {
    uint16_t address = PC + (int8_t)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BPL, `{
  if(!N)
  {
    uint16_t address = PC + (int8_t)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BVC, `{
  if(!V)
  {
    uint16_t address = PC + (int8_t)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')

define(M6502_BVS, `{
  if(V)
  {
    uint16_t address = PC + (int8_t)operand;
    mySystem->incrementCycles(NOTSAMEPAGE(PC, address) ?
        mySystemCyclesPerProcessorCycle << 1 : mySystemCyclesPerProcessorCycle);
    PC = address;
  }
}')
