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
// $Id: M6502.cxx,v 1.21 2007/01/01 18:04:50 stephena Exp $
//============================================================================

#include "ale/emucore/M6502.hxx"

#include <mutex>
#include <cstdint>
#include <iostream>

static std::once_flag bcd_table_init_once;

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6502::M6502(uint32_t systemCyclesPerProcessorCycle)
    : myExecutionStatus(0),
      mySystem(0),
      mySystemCyclesPerProcessorCycle(systemCyclesPerProcessorCycle)
{
  // Compute the BCD lookup table
  std::call_once(bcd_table_init_once, []() {
    for(uint16_t t = 0; t < 256; ++t)
    {
      ourBCDTable[0][t] = ((t >> 4) * 10) + (t & 0x0f);
      ourBCDTable[1][t] = (((t % 100) / 10) << 4) | (t % 10);
    }
  });

  // Compute the System Cycle table
  for(uint16_t t = 0; t < 256; ++t)
  {
    myInstructionSystemCycleTable[t] = ourInstructionProcessorCycleTable[t] *
        mySystemCyclesPerProcessorCycle;
  }

  myTotalInstructionCount = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6502::~M6502()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6502::install(System& system)
{
  // Remember which system I'm installed in
  mySystem = &system;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6502::reset()
{
  // Clear the execution status flags
  myExecutionStatus = 0;

  // Set registers to default values
  A = X = Y = 0;
  SP = 0xff;
  PS(0x20);

  // Reset access flag
  myLastAccessWasRead = true;

  // Load PC from the reset vector
  PC = (uint16_t)mySystem->peek(0xfffc) | ((uint16_t)mySystem->peek(0xfffd) << 8);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6502::irq()
{
  myExecutionStatus |= MaskableInterruptBit;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6502::nmi()
{
  myExecutionStatus |= NonmaskableInterruptBit;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6502::stop()
{
  myExecutionStatus |= StopExecutionBit;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6502::AddressingMode M6502::addressingMode(uint8_t opcode) const
{
  return ourAddressingModeTable[opcode];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t M6502::PS() const
{
  uint8_t ps = 0x20;

  if(N)
    ps |= 0x80;
  if(V)
    ps |= 0x40;
  if(B)
    ps |= 0x10;
  if(D)
    ps |= 0x08;
  if(I)
    ps |= 0x04;
  if(!notZ)
    ps |= 0x02;
  if(C)
    ps |= 0x01;

  return ps;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6502::PS(uint8_t ps)
{
  N = ps & 0x80;
  V = ps & 0x40;
  B = true;        // B = ps & 0x10;  The 6507's B flag always true
  D = ps & 0x08;
  I = ps & 0x04;
  notZ = !(ps & 0x02);
  C = ps & 0x01;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
std::ostream& operator<<(std::ostream& out, const M6502::AddressingMode& mode)
{
  switch(mode)
  {
    case M6502::Absolute:
      out << "$nnnn  ";
      break;
    case M6502::AbsoluteX:
      out << "$nnnn,X";
      break;
    case M6502::AbsoluteY:
      out << "$nnnn,Y";
      break;
    case M6502::Implied:
      out << "implied";
      break;
    case M6502::Immediate:
      out << "#$nn   ";
      break;
    case M6502::Indirect:
      out << "($nnnn)";
      break;
    case M6502::IndirectX:
      out << "($nn,X)";
      break;
    case M6502::IndirectY:
      out << "($nn),Y";
      break;
    case M6502::Invalid:
      out << "invalid";
      break;
    case M6502::Relative:
      out << "$nn    ";
      break;
    case M6502::Zero:
      out << "$nn    ";
      break;
    case M6502::ZeroX:
      out << "$nn,X  ";
      break;
    case M6502::ZeroY:
      out << "$nn,Y  ";
      break;
  }
  return out;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t M6502::ourBCDTable[2][256];

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6502::AddressingMode M6502::ourAddressingModeTable[256] = {
    Implied,    IndirectX, Invalid,   IndirectX,    // 0x0?
    Zero,       Zero,      Zero,      Zero,
    Implied,    Immediate, Implied,   Immediate,
    Absolute,   Absolute,  Absolute,  Absolute,

    Relative,   IndirectY, Invalid,   IndirectY,    // 0x1?
    ZeroX,      ZeroX,     ZeroX,     ZeroX,
    Implied,    AbsoluteY, Implied,   AbsoluteY,
    AbsoluteX,  AbsoluteX, AbsoluteX, AbsoluteX,

    Absolute,   IndirectX, Invalid,   IndirectX,    // 0x2?
    Zero,       Zero,      Zero,      Zero,
    Implied,    Immediate, Implied,   Immediate,
    Absolute,   Absolute,  Absolute,  Absolute,

    Relative,   IndirectY, Invalid,   IndirectY,    // 0x3?
    ZeroX,      ZeroX,     ZeroX,     ZeroX,
    Implied,    AbsoluteY, Implied,   AbsoluteY,
    AbsoluteX,  AbsoluteX, AbsoluteX, AbsoluteX,

    Implied,    IndirectX, Invalid,   IndirectX,    // 0x4?
    Zero,       Zero,      Zero,      Zero,
    Implied,    Immediate, Implied,   Immediate,
    Absolute,   Absolute,  Absolute,  Absolute,

    Relative,   IndirectY, Invalid,   IndirectY,    // 0x5?
    ZeroX,      ZeroX,     ZeroX,     ZeroX,
    Implied,    AbsoluteY, Implied,   AbsoluteY,
    AbsoluteX,  AbsoluteX, AbsoluteX, AbsoluteX,

    Implied,    IndirectX, Invalid,   IndirectX,    // 0x6?
    Zero,       Zero,      Zero,      Zero,
    Implied,    Immediate, Implied,   Immediate,
    Indirect,   Absolute,  Absolute,  Absolute,

    Relative,   IndirectY, Invalid,   IndirectY,    // 0x7?
    ZeroX,      ZeroX,     ZeroX,     ZeroX,
    Implied,    AbsoluteY, Implied,   AbsoluteY,
    AbsoluteX,  AbsoluteX, AbsoluteX, AbsoluteX,

    Immediate,  IndirectX, Immediate, IndirectX,    // 0x8?
    Zero,       Zero,      Zero,      Zero,
    Implied,    Immediate, Implied,   Immediate,
    Absolute,   Absolute,  Absolute,  Absolute,

    Relative,   IndirectY, Invalid,   IndirectY,    // 0x9?
    ZeroX,      ZeroX,     ZeroY,     ZeroY,
    Implied,    AbsoluteY, Implied,   AbsoluteY,
    AbsoluteX,  AbsoluteX, AbsoluteY, AbsoluteY,

    Immediate,  IndirectX, Immediate, IndirectX,    // 0xA?
    Zero,       Zero,      Zero,      Zero,
    Implied,    Immediate, Implied,   Immediate,
    Absolute,   Absolute,  Absolute,  Absolute,

    Relative,   IndirectY, Invalid,   IndirectY,    // 0xB?
    ZeroX,      ZeroX,     ZeroY,     ZeroY,
    Implied,    AbsoluteY, Implied,   AbsoluteY,
    AbsoluteX,  AbsoluteX, AbsoluteY, AbsoluteY,

    Immediate,  IndirectX, Immediate, IndirectX,    // 0xC?
    Zero,       Zero,      Zero,      Zero,
    Implied,    Immediate, Implied,   Immediate,
    Absolute,   Absolute,  Absolute,  Absolute,

    Relative,   IndirectY, Invalid,   IndirectY,    // 0xD?
    ZeroX,      ZeroX,     ZeroX,     ZeroX,
    Implied,    AbsoluteY, Implied,   AbsoluteY,
    AbsoluteX,  AbsoluteX, AbsoluteX, AbsoluteX,

    Immediate,  IndirectX, Immediate, IndirectX,    // 0xE?
    Zero,       Zero,      Zero,      Zero,
    Implied,    Immediate, Implied,   Immediate,
    Absolute,   Absolute,  Absolute,  Absolute,

    Relative,   IndirectY, Invalid,   IndirectY,    // 0xF?
    ZeroX,      ZeroX,     ZeroX,     ZeroX,
    Implied,    AbsoluteY, Implied,   AbsoluteY,
    AbsoluteX,  AbsoluteX, AbsoluteX, AbsoluteX
  };

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint32_t M6502::ourInstructionProcessorCycleTable[256] = {
//  0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
    7, 6, 2, 8, 3, 3, 5, 5, 3, 2, 2, 2, 4, 4, 6, 6,  // 0
    2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  // 1
    6, 6, 2, 8, 3, 3, 5, 5, 4, 2, 2, 2, 4, 4, 6, 6,  // 2
    2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  // 3
    6, 6, 2, 8, 3, 3, 5, 5, 3, 2, 2, 2, 3, 4, 6, 6,  // 4
    2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  // 5
    6, 6, 2, 8, 3, 3, 5, 5, 4, 2, 2, 2, 5, 4, 6, 6,  // 6
    2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  // 7
    2, 6, 2, 6, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,  // 8
    2, 6, 2, 6, 4, 4, 4, 4, 2, 5, 2, 5, 5, 5, 5, 5,  // 9
    2, 6, 2, 6, 3, 3, 3, 4, 2, 2, 2, 2, 4, 4, 4, 4,  // a
    2, 5, 2, 5, 4, 4, 4, 4, 2, 4, 2, 4, 4, 4, 4, 4,  // b
    2, 6, 2, 8, 3, 3, 5, 5, 2, 2, 2, 2, 4, 4, 6, 6,  // c
    2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7,  // d
    2, 6, 2, 8, 3, 3, 5, 5, 2, 2, 2, 2, 4, 4, 6, 6,  // e
    2, 5, 2, 8, 4, 4, 6, 6, 2, 4, 2, 7, 4, 4, 7, 7   // f
  };

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* M6502::ourInstructionMnemonicTable[256] = {
  "BRK",  "ORA",  "n/a",  "slo",  "nop",  "ORA",  "ASL",  "slo",    // 0x0?
  "PHP",  "ORA",  "ASLA", "anc",  "nop",  "ORA",  "ASL",  "slo",

  "BPL",  "ORA",  "n/a",  "slo",  "nop",  "ORA",  "ASL",  "slo",    // 0x1?
  "CLC",  "ORA",  "nop",  "slo",  "nop",  "ORA",  "ASL",  "slo",

  "JSR",  "AND",  "n/a",  "rla",  "BIT",  "AND",  "ROL",  "rla",    // 0x2?
  "PLP",  "AND",  "ROLA", "anc",  "BIT",  "AND",  "ROL",  "rla",

  "BMI",  "AND",  "n/a",  "rla",  "nop",  "AND",  "ROL",  "rla",    // 0x3?
  "SEC",  "AND",  "nop",  "rla",  "nop",  "AND",  "ROL",  "rla",

  "RTI",  "EOR",  "n/a",  "sre",  "nop",  "EOR",  "LSR",  "sre",    // 0x4?
  "PHA",  "EOR",  "LSRA", "asr",  "JMP",  "EOR",  "LSR",  "sre",

  "BVC",  "EOR",  "n/a",  "sre",  "nop",  "EOR",  "LSR",  "sre",    // 0x5?
  "CLI",  "EOR",  "nop",  "sre",  "nop",  "EOR",  "LSR",  "sre",

  "RTS",  "ADC",  "n/a",  "rra",  "nop",  "ADC",  "ROR",  "rra",    // 0x6?
  "PLA",  "ADC",  "RORA", "arr",  "JMP",  "ADC",  "ROR",  "rra",

  "BVS",  "ADC",  "n/a",  "rra",  "nop",  "ADC",  "ROR",  "rra",    // 0x7?
  "SEI",  "ADC",  "nop",  "rra",  "nop",  "ADC",  "ROR",  "rra",

  "nop",  "STA",  "nop",  "sax",  "STY",  "STA",  "STX",  "sax",    // 0x8?
  "DEY",  "nop",  "TXA",  "ane",  "STY",  "STA",  "STX",  "sax",

  "BCC",  "STA",  "n/a",  "sha",  "STY",  "STA",  "STX",  "sax",    // 0x9?
  "TYA",  "STA",  "TXS",  "shs",  "shy",  "STA",  "shx",  "sha",

  "LDY",  "LDA",  "LDX",  "lax",  "LDY",  "LDA",  "LDX",  "lax",    // 0xA?
  "TAY",  "LDA",  "TAX",  "lxa",  "LDY",  "LDA",  "LDX",  "lax",

  "BCS",  "LDA",  "n/a",  "lax",  "LDY",  "LDA",  "LDX",  "lax",    // 0xB?
  "CLV",  "LDA",  "TSX",  "las",  "LDY",  "LDA",  "LDX",  "lax",

  "CPY",  "CMP",  "nop",  "dcp",  "CPY",  "CMP",  "DEC",  "dcp",    // 0xC?
  "INY",  "CMP",  "DEX",  "sbx",  "CPY",  "CMP",  "DEC",  "dcp",

  "BNE",  "CMP",  "n/a",  "dcp",  "nop",  "CMP",  "DEC",  "dcp",    // 0xD?
  "CLD",  "CMP",  "nop",  "dcp",  "nop",  "CMP",  "DEC",  "dcp",

  "CPX",  "SBC",  "nop",  "isb",  "CPX",  "SBC",  "INC",  "isb",    // 0xE?
  "INX",  "SBC",  "NOP",  "sbc",  "CPX",  "SBC",  "INC",  "isb",

  "BEQ",  "SBC",  "n/a",  "isb",  "nop",  "SBC",  "INC",  "isb",    // 0xF?
  "SED",  "SBC",  "nop",  "isb",  "nop",  "SBC",  "INC",  "isb"
};

}  // namespace stella
}  // namespace ale
