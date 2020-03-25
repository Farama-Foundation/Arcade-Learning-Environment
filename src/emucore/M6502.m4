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
// $Id: M6502.m4,v 1.4 2005/06/16 01:11:28 stephena Exp $
//============================================================================

/** 
  Code and cases to emulate each of the 6502 instruction 

  @author  Bradford W. Mott
  @version $Id: M6502.m4,v 1.4 2005/06/16 01:11:28 stephena Exp $
*/

#ifndef NOTSAMEPAGE
  #define NOTSAMEPAGE(_addr1, _addr2) (((_addr1) ^ (_addr2)) & 0xff00)
#endif

define(M6502_ADC, `{
  uInt8 oldA = A;

  if(!D)
  {
    Int16 sum = (Int16)((Int8)A) + (Int16)((Int8)operand) + (C ? 1 : 0);
    V = ((sum > 127) || (sum < -128));

    sum = (Int16)A + (Int16)operand + (C ? 1 : 0);
    A = sum;
    C = (sum > 0xff);
    notZ = A;
    N = A & 0x80;
  }
  else
  {
    Int16 sum = ourBCDTable[0][A] + ourBCDTable[0][operand] + (C ? 1 : 0);

    C = (sum > 99);
    A = ourBCDTable[1][sum & 0xff];
    notZ = A;
    N = A & 0x80;
    V = ((oldA ^ A) & 0x80) && ((A ^ operand) & 0x80);
  }
}')

define(M6502_ANC, `{
  A &= operand;
  notZ = A;
  N = A & 0x80;
  C = N;
}')

define(M6502_AND, `{
  A &= operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_ANE, `{
  // NOTE: The implementation of this instruction is based on
  // information from the 64doc.txt file.  This instruction is
  // reported to be unstable!
  A = (A | 0xee) & X & operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_ARR, `{
  // NOTE: The implementation of this instruction is based on
  // information from the 64doc.txt file.  There are mixed
  // reports on its operation!
  if(!D)
  {
    A &= operand;
    A = ((A >> 1) & 0x7f) | (C ? 0x80 : 0x00);

    C = A & 0x40;
    V = (A & 0x40) ^ ((A & 0x20) << 1);

    notZ = A;
    N = A & 0x80;
  }
  else
  {
    uInt8 value = A & operand;

    A = ((value >> 1) & 0x7f) | (C ? 0x80 : 0x00);
    N = C;
    notZ = A;
    V = (value ^ A) & 0x40;

    if(((value & 0x0f) + (value & 0x01)) > 0x05)
    {
      A = (A & 0xf0) | ((A + 0x06) & 0x0f);
    }
    
    if(((value & 0xf0) + (value & 0x10)) > 0x50) 
    {
      A = (A + 0x60) & 0xff;
      C = 1;
    }
    else
    {
      C = 0;
    }
  }
}')

define(M6502_ASL, `{
  // Set carry flag according to the left-most bit in value
  C = operand & 0x80;

  operand <<= 1;
  poke(operandAddress, operand);

  notZ = operand;
  N = operand & 0x80;
}')

define(M6502_ASLA, `{
  // Set carry flag according to the left-most bit in A
  C = A & 0x80;

  A <<= 1;

  notZ = A;
  N = A & 0x80;
}')

define(M6502_ASR, `{
  A &= operand;

  // Set carry flag according to the right-most bit
  C = A & 0x01;

  A = (A >> 1) & 0x7f;

  notZ = A;
  N = A & 0x80;
}')

define(M6502_BIT, `{
  notZ = (A & operand);
  N = operand & 0x80;
  V = operand & 0x40;
}')

define(M6502_BRK, `{
  peek(PC++);

  B = true;

  poke(0x0100 + SP--, PC >> 8);
  poke(0x0100 + SP--, PC & 0x00ff);
  poke(0x0100 + SP--, PS());

  I = true;

  PC = peek(0xfffe);
  PC |= ((uInt16)peek(0xffff) << 8);
}')

define(M6502_CLC, `{
  C = false;
}')

define(M6502_CLD, `{
  D = false;
}')

define(M6502_CLI, `{
  I = false;
}')

define(M6502_CLV, `{
  V = false;
}')

define(M6502_CMP, `{
  uInt16 value = (uInt16)A - (uInt16)operand;

  notZ = value;
  N = value & 0x0080;
  C = !(value & 0x0100);
}')

define(M6502_CPX, `{
  uInt16 value = (uInt16)X - (uInt16)operand;

  notZ = value;
  N = value & 0x0080;
  C = !(value & 0x0100);
}')

define(M6502_CPY, `{
  uInt16 value = (uInt16)Y - (uInt16)operand;

  notZ = value;
  N = value & 0x0080;
  C = !(value & 0x0100);
}')

define(M6502_DCP, `{
  uInt8 value = operand - 1;
  poke(operandAddress, value);

  uInt16 value2 = (uInt16)A - (uInt16)value;
  notZ = value2;
  N = value2 & 0x0080;
  C = !(value2 & 0x0100);
}')

define(M6502_DEC, `{
  uInt8 value = operand - 1;
  poke(operandAddress, value);

  notZ = value;
  N = value & 0x80;
}')

define(M6502_DEX, `{
  X--;

  notZ = X;
  N = X & 0x80;
}')


define(M6502_DEY, `{
  Y--;

  notZ = Y;
  N = Y & 0x80;
}')

define(M6502_EOR, `{
  A ^= operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_INC, `{
  uInt8 value = operand + 1;
  poke(operandAddress, value);

  notZ = value;
  N = value & 0x80;
}')

define(M6502_INX, `{
  X++;
  notZ = X;
  N = X & 0x80;
}')

define(M6502_INY, `{
  Y++;
  notZ = Y;
  N = Y & 0x80;
}')

define(M6502_ISB, `{
  operand = operand + 1;
  poke(operandAddress, operand);

  uInt8 oldA = A;

  if(!D)
  {
    operand = ~operand;
    Int16 difference = (Int16)((Int8)A) + (Int16)((Int8)operand) + (C ? 1 : 0);
    V = ((difference > 127) || (difference < -128));

    difference = ((Int16)A) + ((Int16)operand) + (C ? 1 : 0);
    A = difference;
    C = (difference > 0xff);
    notZ = A;
    N = A & 0x80;
  }
  else
  {
    Int16 difference = ourBCDTable[0][A] - ourBCDTable[0][operand] 
        - (C ? 0 : 1);

    if(difference < 0)
      difference += 100;

    A = ourBCDTable[1][difference];
    notZ = A;
    N = A & 0x80;

    C = (oldA >= (operand + (C ? 0 : 1)));
    V = ((oldA ^ A) & 0x80) && ((A ^ operand) & 0x80);
  }
}')

define(M6502_JMP, `{
  PC = operandAddress;
}')

define(M6502_JSR, `{
  uInt8 low = peek(PC++);
  peek(0x0100 + SP);

  // It seems that the 650x does not push the address of the next instruction
  // on the stack it actually pushes the address of the next instruction
  // minus one.  This is compensated for in the RTS instruction
  poke(0x0100 + SP--, PC >> 8);
  poke(0x0100 + SP--, PC & 0xff);

  PC = low | ((uInt16)peek(PC++) << 8); 
}')

define(M6502_LAS, `{
  A = X = SP = SP & operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_LAX, `{
  A = operand;
  X = operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_LDA, `{
  A = operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_LDX, `{
  X = operand;
  notZ = X;
  N = X & 0x80;
}')

define(M6502_LDY, `{
  Y = operand;
  notZ = Y;
  N = Y & 0x80;
}')

define(M6502_LSR, `{
  // Set carry flag according to the right-most bit in value
  C = operand & 0x01;

  operand = (operand >> 1) & 0x7f;
  poke(operandAddress, operand);

  notZ = operand;
  N = operand & 0x80;
}')

define(M6502_LSRA, `{
  // Set carry flag according to the right-most bit
  C = A & 0x01;

  A = (A >> 1) & 0x7f;

  notZ = A;
  N = A & 0x80;
}')

define(M6502_LXA, `{
  // NOTE: The implementation of this instruction is based on
  // information from the 64doc.txt file.  This instruction is
  // reported to be very unstable!
  A = X = (A | 0xee) & operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_NOP, `{
}')

define(M6502_ORA, `{
  A |= operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_PHA, `{
  poke(0x0100 + SP--, A);
}')

define(M6502_PHP, `{
  poke(0x0100 + SP--, PS());
}')

define(M6502_PLA, `{
  peek(0x0100 + SP++);
  A = peek(0x0100 + SP);
  notZ = A;
  N = A & 0x80;
}')

define(M6502_PLP, `{
  peek(0x0100 + SP++);
  PS(peek(0x0100 + SP));
}')

define(M6502_RLA, `{
  uInt8 value = (operand << 1) | (C ? 1 : 0);
  poke(operandAddress, value);

  A &= value;
  C = operand & 0x80;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_ROL, `{
  bool oldC = C;

  // Set carry flag according to the left-most bit in operand
  C = operand & 0x80;

  operand = (operand << 1) | (oldC ? 1 : 0);
  poke(operandAddress, operand);

  notZ = operand;
  N = operand & 0x80;
}')

define(M6502_ROLA, `{
  bool oldC = C;

  // Set carry flag according to the left-most bit
  C = A & 0x80;

  A = (A << 1) | (oldC ? 1 : 0);

  notZ = A;
  N = A & 0x80;
}')

define(M6502_ROR, `{
  bool oldC = C;

  // Set carry flag according to the right-most bit
  C = operand & 0x01;

  operand = ((operand >> 1) & 0x7f) | (oldC ? 0x80 : 0x00);
  poke(operandAddress, operand);

  notZ = operand;
  N = operand & 0x80;
}')

define(M6502_RORA, `{
  bool oldC = C;

  // Set carry flag according to the right-most bit
  C = A & 0x01;

  A = ((A >> 1) & 0x7f) | (oldC ? 0x80 : 0x00);

  notZ = A;
  N = A & 0x80;
}')

define(M6502_RRA, `{
  uInt8 oldA = A;
  bool oldC = C;

  // Set carry flag according to the right-most bit
  C = operand & 0x01;

  operand = ((operand >> 1) & 0x7f) | (oldC ? 0x80 : 0x00);
  poke(operandAddress, operand);

  if(!D)
  {
    Int16 sum = (Int16)((Int8)A) + (Int16)((Int8)operand) + (C ? 1 : 0);
    V = ((sum > 127) || (sum < -128));

    sum = (Int16)A + (Int16)operand + (C ? 1 : 0);
    A = sum;
    C = (sum > 0xff);
    notZ = A;
    N = A & 0x80;
  }
  else
  {
    Int16 sum = ourBCDTable[0][A] + ourBCDTable[0][operand] + (C ? 1 : 0);

    C = (sum > 99);
    A = ourBCDTable[1][sum & 0xff];
    notZ = A;
    N = A & 0x80;
    V = ((oldA ^ A) & 0x80) && ((A ^ operand) & 0x80);
  }
}')

define(M6502_RTI, `{
  peek(0x0100 + SP++);
  PS(peek(0x0100 + SP++));
  PC = peek(0x0100 + SP++);
  PC |= ((uInt16)peek(0x0100 + SP) << 8);
}')

define(M6502_RTS, `{
  peek(0x0100 + SP++);
  PC = peek(0x0100 + SP++);
  PC |= ((uInt16)peek(0x0100 + SP) << 8);
  peek(PC++);
}')

define(M6502_SAX, `{
  poke(operandAddress, A & X);
}')

define(M6502_SBC, `{
  uInt8 oldA = A;

  if(!D)
  {
    operand = ~operand;
    Int16 difference = (Int16)((Int8)A) + (Int16)((Int8)operand) + (C ? 1 : 0);
    V = ((difference > 127) || (difference < -128));

    difference = ((Int16)A) + ((Int16)operand) + (C ? 1 : 0);
    A = difference;
    C = (difference > 0xff);
    notZ = A;
    N = A & 0x80;
  }
  else
  {
    Int16 difference = ourBCDTable[0][A] - ourBCDTable[0][operand] 
        - (C ? 0 : 1);

    if(difference < 0)
      difference += 100;

    A = ourBCDTable[1][difference];
    notZ = A;
    N = A & 0x80;

    C = (oldA >= (operand + (C ? 0 : 1)));
    V = ((oldA ^ A) & 0x80) && ((A ^ operand) & 0x80);
  }
}')

define(M6502_SBX, `{
  uInt16 value = (uInt16)(X & A) - (uInt16)operand;
  X = (value & 0xff);

  notZ = X;
  N = X & 0x80;
  C = !(value & 0x0100);
}')

define(M6502_SEC, `{
  C = true;
}')

define(M6502_SED, `{
  D = true;
}')

define(M6502_SEI, `{
  I = true;
}')

define(M6502_SHA, `{
  // NOTE: There are mixed reports on the actual operation
  // of this instruction!
  poke(operandAddress, A & X & (((operandAddress >> 8) & 0xff) + 1)); 
}')

define(M6502_SHS, `{
  // NOTE: There are mixed reports on the actual operation
  // of this instruction!
  SP = A & X;
  poke(operandAddress, A & X & (((operandAddress >> 8) & 0xff) + 1)); 
}')

define(M6502_SHX, `{
  // NOTE: There are mixed reports on the actual operation
  // of this instruction!
  poke(operandAddress, X & (((operandAddress >> 8) & 0xff) + 1)); 
}')

define(M6502_SHY, `{
  // NOTE: There are mixed reports on the actual operation
  // of this instruction!
  poke(operandAddress, Y & (((operandAddress >> 8) & 0xff) + 1)); 
}')

define(M6502_SLO, `{
  // Set carry flag according to the left-most bit in value
  C = operand & 0x80;

  operand <<= 1;
  poke(operandAddress, operand);

  A |= operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_SRE, `{
  // Set carry flag according to the right-most bit in value
  C = operand & 0x01;

  operand = (operand >> 1) & 0x7f;
  poke(operandAddress, operand);

  A ^= operand;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_STA, `{
  poke(operandAddress, A);
}')

define(M6502_STX, `{
  poke(operandAddress, X);
}')

define(M6502_STY, `{
  poke(operandAddress, Y);
}')

define(M6502_TAX, `{
  X = A;
  notZ = X;
  N = X & 0x80;
}')

define(M6502_TAY, `{
  Y = A;
  notZ = Y;
  N = Y & 0x80;
}')

define(M6502_TSX, `{
  X = SP;
  notZ = X;
  N = X & 0x80;
}')

define(M6502_TXA, `{
  A = X;
  notZ = A;
  N = A & 0x80;
}')

define(M6502_TXS, `{
  SP = X;
}')

define(M6502_TYA, `{
  A = Y;
  notZ = A;
  N = A & 0x80;
}')


case 0x69:
M6502_IMMEDIATE_READ
M6502_ADC
break;

case 0x65:
M6502_ZERO_READ
M6502_ADC
break;

case 0x75:
M6502_ZEROX_READ
M6502_ADC
break;

case 0x6d:
M6502_ABSOLUTE_READ
M6502_ADC
break;

case 0x7d:
M6502_ABSOLUTEX_READ
M6502_ADC
break;

case 0x79:
M6502_ABSOLUTEY_READ
M6502_ADC
break;

case 0x61:
M6502_INDIRECTX_READ
M6502_ADC
break;

case 0x71:
M6502_INDIRECTY_READ
M6502_ADC
break;


case 0x4b:
M6502_IMMEDIATE_READ
M6502_ASR
break;


case 0x0b:
case 0x2b:
M6502_IMMEDIATE_READ
M6502_ANC
break;


case 0x29:
M6502_IMMEDIATE_READ
M6502_AND
break;

case 0x25:
M6502_ZERO_READ
M6502_AND
break;

case 0x35:
M6502_ZEROX_READ
M6502_AND
break;

case 0x2d:
M6502_ABSOLUTE_READ
M6502_AND
break;

case 0x3d:
M6502_ABSOLUTEX_READ
M6502_AND
break;

case 0x39:
M6502_ABSOLUTEY_READ
M6502_AND
break;

case 0x21:
M6502_INDIRECTX_READ
M6502_AND
break;

case 0x31:
M6502_INDIRECTY_READ
M6502_AND
break;


case 0x8b:
M6502_IMMEDIATE_READ
M6502_ANE
break;


case 0x6b:
M6502_IMMEDIATE_READ
M6502_ARR
break;


case 0x0a:
M6502_IMPLIED
M6502_ASLA
break;

case 0x06:
M6502_ZERO_READMODIFYWRITE
M6502_ASL
break;

case 0x16:
M6502_ZEROX_READMODIFYWRITE
M6502_ASL
break;

case 0x0e:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_ASL
break;

case 0x1e:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_ASL
break;


case 0x90:
M6502_IMMEDIATE_READ
M6502_BCC
break;


case 0xb0:
M6502_IMMEDIATE_READ
M6502_BCS
break;


case 0xf0:
M6502_IMMEDIATE_READ
M6502_BEQ
break;


case 0x24:
M6502_ZERO_READ
M6502_BIT
break;

case 0x2C:
M6502_ABSOLUTE_READ
M6502_BIT
break;


case 0x30:
M6502_IMMEDIATE_READ
M6502_BMI
break;


case 0xD0:
M6502_IMMEDIATE_READ
M6502_BNE
break;


case 0x10:
M6502_IMMEDIATE_READ
M6502_BPL
break;


case 0x00:
M6502_BRK
break;


case 0x50:
M6502_IMMEDIATE_READ
M6502_BVC
break;


case 0x70:
M6502_IMMEDIATE_READ
M6502_BVS
break;


case 0x18:
M6502_IMPLIED
M6502_CLC
break;


case 0xd8:
M6502_IMPLIED
M6502_CLD
break;


case 0x58:
M6502_IMPLIED
M6502_CLI
break;


case 0xb8:
M6502_IMPLIED
M6502_CLV
break;


case 0xc9:
M6502_IMMEDIATE_READ
M6502_CMP
break;

case 0xc5:
M6502_ZERO_READ
M6502_CMP
break;

case 0xd5:
M6502_ZEROX_READ
M6502_CMP
break;

case 0xcd:
M6502_ABSOLUTE_READ
M6502_CMP
break;

case 0xdd:
M6502_ABSOLUTEX_READ
M6502_CMP
break;

case 0xd9:
M6502_ABSOLUTEY_READ
M6502_CMP
break;

case 0xc1:
M6502_INDIRECTX_READ
M6502_CMP
break;

case 0xd1:
M6502_INDIRECTY_READ
M6502_CMP
break;


case 0xe0:
M6502_IMMEDIATE_READ
M6502_CPX
break;

case 0xe4:
M6502_ZERO_READ
M6502_CPX
break;

case 0xec:
M6502_ABSOLUTE_READ
M6502_CPX
break;


case 0xc0:
M6502_IMMEDIATE_READ
M6502_CPY
break;

case 0xc4:
M6502_ZERO_READ
M6502_CPY
break;

case 0xcc:
M6502_ABSOLUTE_READ
M6502_CPY
break;


case 0xcf:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_DCP
break;

case 0xdf:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_DCP
break;

case 0xdb:
M6502_ABSOLUTEY_READMODIFYWRITE
M6502_DCP
break;

case 0xc7:
M6502_ZERO_READMODIFYWRITE
M6502_DCP
break;

case 0xd7:
M6502_ZEROX_READMODIFYWRITE
M6502_DCP
break;

case 0xc3:
M6502_INDIRECTX_READMODIFYWRITE
M6502_DCP
break;

case 0xd3:
M6502_INDIRECTY_READMODIFYWRITE
M6502_DCP
break;


case 0xc6:
M6502_ZERO_READMODIFYWRITE
M6502_DEC
break;

case 0xd6:
M6502_ZEROX_READMODIFYWRITE
M6502_DEC
break;

case 0xce:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_DEC
break;

case 0xde:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_DEC
break;


case 0xca:
M6502_IMPLIED
M6502_DEX
break;


case 0x88:
M6502_IMPLIED
M6502_DEY
break;


case 0x49:
M6502_IMMEDIATE_READ
M6502_EOR
break;

case 0x45:
M6502_ZERO_READ
M6502_EOR
break;

case 0x55:
M6502_ZEROX_READ
M6502_EOR
break;

case 0x4d:
M6502_ABSOLUTE_READ
M6502_EOR
break;

case 0x5d:
M6502_ABSOLUTEX_READ
M6502_EOR
break;

case 0x59:
M6502_ABSOLUTEY_READ
M6502_EOR
break;

case 0x41:
M6502_INDIRECTX_READ
M6502_EOR
break;

case 0x51:
M6502_INDIRECTY_READ
M6502_EOR
break;


case 0xe6:
M6502_ZERO_READMODIFYWRITE
M6502_INC
break;

case 0xf6:
M6502_ZEROX_READMODIFYWRITE
M6502_INC
break;

case 0xee:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_INC
break;

case 0xfe:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_INC
break;


case 0xe8:
M6502_IMPLIED
M6502_INX
break;


case 0xc8:
M6502_IMPLIED
M6502_INY
break;


case 0xef:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_ISB
break;

case 0xff:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_ISB
break;

case 0xfb:
M6502_ABSOLUTEY_READMODIFYWRITE
M6502_ISB
break;

case 0xe7:
M6502_ZERO_READMODIFYWRITE
M6502_ISB
break;

case 0xf7:
M6502_ZEROX_READMODIFYWRITE
M6502_ISB
break;

case 0xe3:
M6502_INDIRECTX_READMODIFYWRITE
M6502_ISB
break;

case 0xf3:
M6502_INDIRECTY_READMODIFYWRITE
M6502_ISB
break;


case 0x4c:
M6502_ABSOLUTE_WRITE
M6502_JMP
break;

case 0x6c:
M6502_INDIRECT
M6502_JMP
break;


case 0x20:
M6502_JSR
break;


case 0xbb:
M6502_ABSOLUTEY_READ
M6502_LAS
break;


case 0xaf:
M6502_ABSOLUTE_READ
M6502_LAX
break;

case 0xbf:
M6502_ABSOLUTEY_READ
M6502_LAX
break;

case 0xa7:
M6502_ZERO_READ
M6502_LAX
break;

case 0xb7:
M6502_ZEROY_READ
M6502_LAX
break;

case 0xa3:
M6502_INDIRECTX_READ
M6502_LAX
break;

case 0xb3:
M6502_INDIRECTY_READ
M6502_LAX
break;


case 0xa9:
M6502_IMMEDIATE_READ
M6502_LDA
break;

case 0xa5:
M6502_ZERO_READ
M6502_LDA
break;

case 0xb5:
M6502_ZEROX_READ
M6502_LDA
break;

case 0xad:
M6502_ABSOLUTE_READ
M6502_LDA
break;

case 0xbd:
M6502_ABSOLUTEX_READ
M6502_LDA
break;

case 0xb9:
M6502_ABSOLUTEY_READ
M6502_LDA
break;

case 0xa1:
M6502_INDIRECTX_READ
M6502_LDA
break;

case 0xb1:
M6502_INDIRECTY_READ
M6502_LDA
break;


case 0xa2:
M6502_IMMEDIATE_READ
M6502_LDX
break;

case 0xa6:
M6502_ZERO_READ
M6502_LDX
break;

case 0xb6:
M6502_ZEROY_READ
M6502_LDX
break;

case 0xae:
M6502_ABSOLUTE_READ
M6502_LDX
break;

case 0xbe:
M6502_ABSOLUTEY_READ
M6502_LDX
break;


case 0xa0:
M6502_IMMEDIATE_READ
M6502_LDY
break;

case 0xa4:
M6502_ZERO_READ
M6502_LDY
break;

case 0xb4:
M6502_ZEROX_READ
M6502_LDY
break;

case 0xac:
M6502_ABSOLUTE_READ
M6502_LDY
break;

case 0xbc:
M6502_ABSOLUTEX_READ
M6502_LDY
break;


case 0x4a:
M6502_IMPLIED
M6502_LSRA
break;


case 0x46:
M6502_ZERO_READMODIFYWRITE
M6502_LSR
break;

case 0x56:
M6502_ZEROX_READMODIFYWRITE
M6502_LSR
break;

case 0x4e:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_LSR
break;

case 0x5e:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_LSR
break;


case 0xab:
M6502_IMMEDIATE_READ
M6502_LXA
break;


case 0x1a:
case 0x3a:
case 0x5a:
case 0x7a:
case 0xda:
case 0xea:
case 0xfa:
M6502_IMPLIED
M6502_NOP
break;

case 0x80:
case 0x82:
case 0x89:
case 0xc2:
case 0xe2:
M6502_IMMEDIATE_READ
M6502_NOP
break;

case 0x04:
case 0x44:
case 0x64:
M6502_ZERO_READ
M6502_NOP
break;

case 0x14:
case 0x34:
case 0x54:
case 0x74:
case 0xd4:
case 0xf4:
M6502_ZEROX_READ
M6502_NOP
break;

case 0x0c:
M6502_ABSOLUTE_READ
M6502_NOP
break;

case 0x1c:
case 0x3c:
case 0x5c:
case 0x7c:
case 0xdc:
case 0xfc:
M6502_ABSOLUTEX_READ
M6502_NOP
break;


case 0x09:
M6502_IMMEDIATE_READ
M6502_ORA
break;

case 0x05:
M6502_ZERO_READ
M6502_ORA
break;

case 0x15:
M6502_ZEROX_READ
M6502_ORA
break;

case 0x0d:
M6502_ABSOLUTE_READ
M6502_ORA
break;

case 0x1d:
M6502_ABSOLUTEX_READ
M6502_ORA
break;

case 0x19:
M6502_ABSOLUTEY_READ
M6502_ORA
break;

case 0x01:
M6502_INDIRECTX_READ
M6502_ORA
break;

case 0x11:
M6502_INDIRECTY_READ
M6502_ORA
break;


case 0x48:
M6502_IMPLIED
M6502_PHA
break;


case 0x08:
M6502_IMPLIED
M6502_PHP
break;


case 0x68:
M6502_IMPLIED
M6502_PLA
break;


case 0x28:
M6502_IMPLIED
M6502_PLP
break;


case 0x2f:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_RLA
break;

case 0x3f:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_RLA
break;

case 0x3b:
M6502_ABSOLUTEY_READMODIFYWRITE
M6502_RLA
break;

case 0x27:
M6502_ZERO_READMODIFYWRITE
M6502_RLA
break;

case 0x37:
M6502_ZEROX_READMODIFYWRITE
M6502_RLA
break;

case 0x23:
M6502_INDIRECTX_READMODIFYWRITE
M6502_RLA
break;

case 0x33:
M6502_INDIRECTY_READMODIFYWRITE
M6502_RLA
break;


case 0x2a:
M6502_IMPLIED
M6502_ROLA
break;


case 0x26:
M6502_ZERO_READMODIFYWRITE
M6502_ROL
break;

case 0x36:
M6502_ZEROX_READMODIFYWRITE
M6502_ROL
break;

case 0x2e:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_ROL
break;

case 0x3e:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_ROL
break;


case 0x6a:
M6502_IMPLIED
M6502_RORA
break;

case 0x66:
M6502_ZERO_READMODIFYWRITE
M6502_ROR
break;

case 0x76:
M6502_ZEROX_READMODIFYWRITE
M6502_ROR
break;

case 0x6e:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_ROR
break;

case 0x7e:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_ROR
break;


case 0x6f:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_RRA
break;

case 0x7f:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_RRA
break;

case 0x7b:
M6502_ABSOLUTEY_READMODIFYWRITE
M6502_RRA
break;

case 0x67:
M6502_ZERO_READMODIFYWRITE
M6502_RRA
break;

case 0x77:
M6502_ZEROX_READMODIFYWRITE
M6502_RRA
break;

case 0x63:
M6502_INDIRECTX_READMODIFYWRITE
M6502_RRA
break;

case 0x73:
M6502_INDIRECTY_READMODIFYWRITE
M6502_RRA
break;


case 0x40:
M6502_IMPLIED
M6502_RTI
break;


case 0x60:
M6502_IMPLIED
M6502_RTS
break;


case 0x8f:
M6502_ABSOLUTE_WRITE
M6502_SAX
break;

case 0x87:
M6502_ZERO_WRITE
M6502_SAX
break;

case 0x97:
M6502_ZEROY_WRITE
M6502_SAX
break;

case 0x83:
M6502_INDIRECTX_WRITE
M6502_SAX
break;


case 0xe9:
case 0xeb:
M6502_IMMEDIATE_READ
M6502_SBC
break;

case 0xe5:
M6502_ZERO_READ
M6502_SBC
break;

case 0xf5:
M6502_ZEROX_READ
M6502_SBC
break;

case 0xed:
M6502_ABSOLUTE_READ
M6502_SBC
break;

case 0xfd:
M6502_ABSOLUTEX_READ
M6502_SBC
break;

case 0xf9:
M6502_ABSOLUTEY_READ
M6502_SBC
break;

case 0xe1:
M6502_INDIRECTX_READ
M6502_SBC
break;

case 0xf1:
M6502_INDIRECTY_READ
M6502_SBC
break;


case 0xcb:
M6502_IMMEDIATE_READ
M6502_SBX
break;


case 0x38:
M6502_IMPLIED
M6502_SEC
break;


case 0xf8:
M6502_IMPLIED
M6502_SED
break;


case 0x78:
M6502_IMPLIED
M6502_SEI
break;


case 0x9f:
M6502_ABSOLUTEY_WRITE
M6502_SHA
break;

case 0x93:
M6502_INDIRECTY_WRITE
M6502_SHA
break;


case 0x9b:
M6502_ABSOLUTEY_WRITE
M6502_SHS
break;


case 0x9e:
M6502_ABSOLUTEY_WRITE
M6502_SHX
break;


case 0x9c:
M6502_ABSOLUTEX_WRITE
M6502_SHY
break;


case 0x0f:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_SLO
break;

case 0x1f:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_SLO
break;

case 0x1b:
M6502_ABSOLUTEY_READMODIFYWRITE
M6502_SLO
break;

case 0x07:
M6502_ZERO_READMODIFYWRITE
M6502_SLO
break;

case 0x17:
M6502_ZEROX_READMODIFYWRITE
M6502_SLO
break;

case 0x03:
M6502_INDIRECTX_READMODIFYWRITE
M6502_SLO
break;

case 0x13:
M6502_INDIRECTY_READMODIFYWRITE
M6502_SLO
break;


case 0x4f:
M6502_ABSOLUTE_READMODIFYWRITE
M6502_SRE
break;

case 0x5f:
M6502_ABSOLUTEX_READMODIFYWRITE
M6502_SRE
break;

case 0x5b:
M6502_ABSOLUTEY_READMODIFYWRITE
M6502_SRE
break;

case 0x47:
M6502_ZERO_READMODIFYWRITE
M6502_SRE
break;

case 0x57:
M6502_ZEROX_READMODIFYWRITE
M6502_SRE
break;

case 0x43:
M6502_INDIRECTX_READMODIFYWRITE
M6502_SRE
break;

case 0x53:
M6502_INDIRECTY_READMODIFYWRITE
M6502_SRE
break;


case 0x85:
M6502_ZERO_WRITE
M6502_STA
break;

case 0x95:
M6502_ZEROX_WRITE
M6502_STA
break;

case 0x8d:
M6502_ABSOLUTE_WRITE
M6502_STA
break;

case 0x9d:
M6502_ABSOLUTEX_WRITE
M6502_STA
break;

case 0x99:
M6502_ABSOLUTEY_WRITE
M6502_STA
break;

case 0x81:
M6502_INDIRECTX_WRITE
M6502_STA
break;

case 0x91:
M6502_INDIRECTY_WRITE
M6502_STA
break;


case 0x86:
M6502_ZERO_WRITE
M6502_STX
break;

case 0x96:
M6502_ZEROY_WRITE
M6502_STX
break;

case 0x8e:
M6502_ABSOLUTE_WRITE
M6502_STX
break;


case 0x84:
M6502_ZERO_WRITE
M6502_STY
break;

case 0x94:
M6502_ZEROX_WRITE
M6502_STY
break;

case 0x8c:
M6502_ABSOLUTE_WRITE
M6502_STY
break;


case 0xaa:
M6502_IMPLIED
M6502_TAX
break;


case 0xa8:
M6502_IMPLIED
M6502_TAY
break;


case 0xba:
M6502_IMPLIED
M6502_TSX
break;


case 0x8a:
M6502_IMPLIED
M6502_TXA
break;


case 0x9a:
M6502_IMPLIED
M6502_TXS
break;


case 0x98:
M6502_IMPLIED
M6502_TYA
break;


