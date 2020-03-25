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
// $Id: M6502Hi.m4,v 1.2 2005/06/16 01:11:29 stephena Exp $
//============================================================================

/**
  Code to handle addressing modes and branch instructions for
  high compatibility emulation

  @author  Bradford W. Mott
  @version $Id: M6502Hi.m4,v 1.2 2005/06/16 01:11:29 stephena Exp $
*/

#ifndef NOTSAMEPAGE
  #define NOTSAMEPAGE(_addr1, _addr2) (((_addr1) ^ (_addr2)) & 0xff00)
#endif

define(M6502_IMPLIED, `{
  peek(PC);
}')

define(M6502_IMMEDIATE_READ, `{
  operand = peek(PC++);
}')

define(M6502_ABSOLUTE_READ, `{
  uInt16 address = peek(PC++);
  address |= ((uInt16)peek(PC++) << 8);
  operand = peek(address);
}')

define(M6502_ABSOLUTE_WRITE, `{
  operandAddress = peek(PC++);
  operandAddress |= ((uInt16)peek(PC++) << 8);
}')

define(M6502_ABSOLUTE_READMODIFYWRITE, `{
  operandAddress = peek(PC++);
  operandAddress |= ((uInt16)peek(PC++) << 8);
  operand = peek(operandAddress);
  poke(operandAddress, operand);
}')

define(M6502_ABSOLUTEX_READ, `{
  uInt16 low = peek(PC++);
  uInt16 high = ((uInt16)peek(PC++) << 8);
  operand = peek(high | (uInt8)(low + X));
  if((low + X) > 0xFF)
    operand = peek((high | low) + X);
}')

define(M6502_ABSOLUTEX_WRITE, `{
  uInt16 low = peek(PC++);
  uInt16 high = ((uInt16)peek(PC++) << 8);
  peek(high | (uInt8)(low + X));
  operandAddress = (high | low) + X;
}')

define(M6502_ABSOLUTEX_READMODIFYWRITE, `{
  uInt16 low = peek(PC++);
  uInt16 high = ((uInt16)peek(PC++) << 8);
  peek(high | (uInt8)(low + X));
  operandAddress = (high | low) + X;
  operand = peek(operandAddress);
  poke(operandAddress, operand);
}')

define(M6502_ABSOLUTEY_READ, `{
  uInt16 low = peek(PC++);
  uInt16 high = ((uInt16)peek(PC++) << 8);
  operand = peek(high | (uInt8)(low + Y));
  if((low + Y) > 0xFF)
    operand = peek((high | low) + Y);
}')

define(M6502_ABSOLUTEY_WRITE, `{
  uInt16 low = peek(PC++);
  uInt16 high = ((uInt16)peek(PC++) << 8);
  peek(high | (uInt8)(low + Y));
  operandAddress = (high | low) + Y;
}')

define(M6502_ABSOLUTEY_READMODIFYWRITE, `{
  uInt16 low = peek(PC++);
  uInt16 high = ((uInt16)peek(PC++) << 8);
  peek(high | (uInt8)(low + Y));
  operandAddress = (high | low) + Y;
  operand = peek(operandAddress);
  poke(operandAddress, operand);
}')

define(M6502_ZERO_READ, `{
  operand = peek(peek(PC++));
}')

define(M6502_ZERO_WRITE, `{
  operandAddress = peek(PC++);
}')

define(M6502_ZERO_READMODIFYWRITE, `{
  operandAddress = peek(PC++);
  operand = peek(operandAddress);
  poke(operandAddress, operand);
}')

define(M6502_ZEROX_READ, `{
  uInt8 address = peek(PC++);
  peek(address);
  address += X;
  operand = peek(address); 
}')

define(M6502_ZEROX_WRITE, `{
  operandAddress = peek(PC++);
  peek(operandAddress);
  operandAddress = (operandAddress + X) & 0xFF;
}')

define(M6502_ZEROX_READMODIFYWRITE, `{
  operandAddress = peek(PC++);
  peek(operandAddress);
  operandAddress = (operandAddress + X) & 0xFF;
  operand = peek(operandAddress);
  poke(operandAddress, operand);
}')

define(M6502_ZEROY_READ, `{
  uInt8 address = peek(PC++);
  peek(address);
  address += Y;
  operand = peek(address); 
}')

define(M6502_ZEROY_WRITE, `{
  operandAddress = peek(PC++);
  peek(operandAddress);
  operandAddress = (operandAddress + Y) & 0xFF;
}')

define(M6502_ZEROY_READMODIFYWRITE, `{
  operandAddress = peek(PC++);
  peek(operandAddress);
  operandAddress = (operandAddress + Y) & 0xFF;
  operand = peek(operandAddress);
  poke(operandAddress, operand);
}')

define(M6502_INDIRECT, `{
  uInt16 addr = peek(PC++);
  addr |= ((uInt16)peek(PC++) << 8);

  // Simulate the error in the indirect addressing mode!
  uInt16 high = NOTSAMEPAGE(addr, addr + 1) ? (addr & 0xff00) : (addr + 1);

  operandAddress = peek(addr);
  operandAddress |= ((uInt16)peek(high) << 8);
}')

define(M6502_INDIRECTX_READ, `{
  uInt8 pointer = peek(PC++);
  peek(pointer);
  pointer += X;
  uInt16 address = peek(pointer++);
  address |= ((uInt16)peek(pointer) << 8);
  operand = peek(address);
}')

define(M6502_INDIRECTX_WRITE, `{
  uInt8 pointer = peek(PC++);
  peek(pointer);
  pointer += X;
  operandAddress = peek(pointer++);
  operandAddress |= ((uInt16)peek(pointer) << 8);
}')

define(M6502_INDIRECTX_READMODIFYWRITE, `{
  uInt8 pointer = peek(PC++);
  peek(pointer);
  pointer += X;
  operandAddress = peek(pointer++);
  operandAddress |= ((uInt16)peek(pointer) << 8);
  operand = peek(operandAddress);
  poke(operandAddress, operand);
}')

define(M6502_INDIRECTY_READ, `{
  uInt8 pointer = peek(PC++);
  uInt16 low = peek(pointer++);
  uInt16 high = ((uInt16)peek(pointer) << 8);
  operand = peek(high | (uInt8)(low + Y));
  if((low + Y) > 0xFF)
    operand = peek((high | low) + Y);
}')

define(M6502_INDIRECTY_WRITE, `{
  uInt8 pointer = peek(PC++);
  uInt16 low = peek(pointer++);
  uInt16 high = ((uInt16)peek(pointer) << 8);
  peek(high | (uInt8)(low + Y));
  operandAddress = (high | low) + Y;
}')

define(M6502_INDIRECTY_READMODIFYWRITE, `{
  uInt8 pointer = peek(PC++);
  uInt16 low = peek(pointer++);
  uInt16 high = ((uInt16)peek(pointer) << 8);
  peek(high | (uInt8)(low + Y));
  operandAddress = (high | low) + Y;
  operand = peek(operandAddress);
  poke(operandAddress, operand);
}')


define(M6502_BCC, `{
  if(!C)
  {
    peek(PC);
    uInt16 address = PC + (Int8)operand;
    if(NOTSAMEPAGE(PC, address))
      peek((PC & 0xFF00) | (address & 0x00FF));
    PC = address;
  }
}')

define(M6502_BCS, `{
  if(C)
  {
    peek(PC);
    uInt16 address = PC + (Int8)operand;
    if(NOTSAMEPAGE(PC, address))
      peek((PC & 0xFF00) | (address & 0x00FF));
    PC = address;
  }
}')

define(M6502_BEQ, `{
  if(!notZ)
  {
    peek(PC);
    uInt16 address = PC + (Int8)operand;
    if(NOTSAMEPAGE(PC, address))
      peek((PC & 0xFF00) | (address & 0x00FF));
    PC = address;
  }
}')

define(M6502_BMI, `{
  if(N)
  {
    peek(PC);
    uInt16 address = PC + (Int8)operand;
    if(NOTSAMEPAGE(PC, address))
      peek((PC & 0xFF00) | (address & 0x00FF));
    PC = address;
  }
}')

define(M6502_BNE, `{
  if(notZ)
  {
    peek(PC);
    uInt16 address = PC + (Int8)operand;
    if(NOTSAMEPAGE(PC, address))
      peek((PC & 0xFF00) | (address & 0x00FF));
    PC = address;
  }
}')

define(M6502_BPL, `{
  if(!N)
  {
    peek(PC);
    uInt16 address = PC + (Int8)operand;
    if(NOTSAMEPAGE(PC, address))
      peek((PC & 0xFF00) | (address & 0x00FF));
    PC = address;
  }
}')

define(M6502_BVC, `{
  if(!V)
  {
    peek(PC);
    uInt16 address = PC + (Int8)operand;
    if(NOTSAMEPAGE(PC, address))
      peek((PC & 0xFF00) | (address & 0x00FF));
    PC = address;
  }
}')

define(M6502_BVS, `{
  if(V)
  {
    peek(PC);
    uInt16 address = PC + (Int8)operand;
    if(NOTSAMEPAGE(PC, address))
      peek((PC & 0xFF00) | (address & 0x00FF));
    PC = address;
  }
}')

