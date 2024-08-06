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
// $Id: M6532.cxx,v 1.10 2007/06/21 12:27:00 stephena Exp $
//============================================================================

#include <iostream>
#include <cassert>

#include "ale/emucore/Console.hxx"
#include "ale/emucore/M6532.hxx"
#include "ale/emucore/Switches.hxx"
#include "ale/emucore/System.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/OSystem.hxx"
#include "ale/common/Log.hpp"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6532::M6532(const Console& console)
    : myConsole(console)
{
  // Randomize the 128 bytes of memory

  for(uint32_t t = 0; t < 128; ++t)
  {
    myRAM[t] = myConsole.system().rng().next();
  }

  // Initialize other data members
  reset();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6532::~M6532()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* M6532::name() const
{
  return "M6532";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6532::reset()
{
  myTimer = 25 + (myConsole.system().rng().next() % 75);
  myIntervalShift = 6;
  myCyclesWhenTimerSet = 0;
  myCyclesWhenInterruptReset = 0;
  myTimerReadAfterInterrupt = false;

  // Zero the I/O registers
  myDDRA = 0x00;
  myDDRB = 0x00;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6532::systemCyclesReset()
{
  // System cycles are being reset to zero so we need to adjust
  // the cycle count we remembered when the timer was last set
  myCyclesWhenTimerSet -= mySystem->cycles();
  myCyclesWhenInterruptReset -= mySystem->cycles();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6532::install(System& system)
{
  // Remember which system I'm installed in
  mySystem = &system;

  uint16_t shift = mySystem->pageShift();
  uint16_t mask = mySystem->pageMask();

  // Make sure the system we're being installed in has a page size that'll work
  assert((0x1080 & mask) == 0);

  // All accesses are to this device
  System::PageAccess access;
  access.device = this;

  // We're installing in a 2600 system
  for(int address = 0; address < 8192; address += (1 << shift))
  {
    if((address & 0x1080) == 0x0080)
    {
      if((address & 0x0200) == 0x0000)
      {
        access.directPeekBase = &myRAM[address & 0x007f];
        access.directPokeBase = &myRAM[address & 0x007f];
        mySystem->setPageAccess(address >> shift, access);
      }
      else
      {
        access.directPeekBase = 0;
        access.directPokeBase = 0;
        mySystem->setPageAccess(address >> shift, access);
      }
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint8_t M6532::peek(uint16_t addr)
{
  switch(addr & 0x07)
  {
    case 0x00:    // Port A I/O Register (Joystick)
    {
      uint8_t value = 0x00;

      if(myConsole.controller(Controller::Left).read(Controller::One))
        value |= 0x10;
      if(myConsole.controller(Controller::Left).read(Controller::Two))
        value |= 0x20;
      if(myConsole.controller(Controller::Left).read(Controller::Three))
        value |= 0x40;
      if(myConsole.controller(Controller::Left).read(Controller::Four))
        value |= 0x80;

      if(myConsole.controller(Controller::Right).read(Controller::One))
        value |= 0x01;
      if(myConsole.controller(Controller::Right).read(Controller::Two))
        value |= 0x02;
      if(myConsole.controller(Controller::Right).read(Controller::Three))
        value |= 0x04;
      if(myConsole.controller(Controller::Right).read(Controller::Four))
        value |= 0x08;

      return value;
    }

    case 0x01:    // Port A Data Direction Register
    {
      return myDDRA;
    }

    case 0x02:    // Port B I/O Register (Console switches)
    {
      return myConsole.switches().read();
    }

    case 0x03:    // Port B Data Direction Register
    {
      return myDDRB;
    }

    case 0x04:    // Timer Output
    case 0x06:
    {
      uint32_t cycles = mySystem->cycles() - 1;
      uint32_t delta = cycles - myCyclesWhenTimerSet;
      int timer = (int)myTimer - (int)(delta >> myIntervalShift) - 1;

      // See if the timer has expired yet?
      if(timer >= 0)
      {
        return (uint8_t)timer;
      }
      else
      {
        timer = (int)(myTimer << myIntervalShift) - (int)delta - 1;

        if((timer <= -2) && !myTimerReadAfterInterrupt)
        {
          // Indicate that timer has been read after interrupt occured
          myTimerReadAfterInterrupt = true;
          myCyclesWhenInterruptReset = mySystem->cycles();
        }

        if(myTimerReadAfterInterrupt)
        {
          int offset = myCyclesWhenInterruptReset -
              (myCyclesWhenTimerSet + (myTimer << myIntervalShift));

          timer = (int)myTimer - (int)(delta >> myIntervalShift) - offset;
        }

        return (uint8_t)timer;
      }
    }

    case 0x05:    // Interrupt Flag
    case 0x07:
    {
      uint32_t cycles = mySystem->cycles() - 1;
      uint32_t delta = cycles - myCyclesWhenTimerSet;
      int timer = (int)myTimer - (int)(delta >> myIntervalShift) - 1;

      if((timer >= 0) || myTimerReadAfterInterrupt)
        return 0x00;
      else
        return 0x80;
    }

    default:
    {
#ifdef DEBUG_ACCESSES
      ale::Logger::Error << "BAD M6532 Peek: " << hex << addr << std::endl;
#endif
      return 0;
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6532::poke(uint16_t addr, uint8_t value)
{
  if((addr & 0x07) == 0x00)         // Port A I/O Register (Joystick)
  {
    uint8_t a = value & myDDRA;

    myConsole.controller(Controller::Left).write(Controller::One, a & 0x10);
    myConsole.controller(Controller::Left).write(Controller::Two, a & 0x20);
    myConsole.controller(Controller::Left).write(Controller::Three, a & 0x40);
    myConsole.controller(Controller::Left).write(Controller::Four, a & 0x80);

    myConsole.controller(Controller::Right).write(Controller::One, a & 0x01);
    myConsole.controller(Controller::Right).write(Controller::Two, a & 0x02);
    myConsole.controller(Controller::Right).write(Controller::Three, a & 0x04);
    myConsole.controller(Controller::Right).write(Controller::Four, a & 0x08);
  }
  else if((addr & 0x07) == 0x01)    // Port A Data Direction Register
  {
    myDDRA = value;
  }
  else if((addr & 0x07) == 0x02)    // Port B I/O Register (Console switches)
  {
    return;
  }
  else if((addr & 0x07) == 0x03)    // Port B Data Direction Register
  {
//        myDDRB = value;
    return;
  }
  else if((addr & 0x17) == 0x14)    // Write timer divide by 1
  {
    myTimer = value;
    myIntervalShift = 0;
    myCyclesWhenTimerSet = mySystem->cycles();
    myTimerReadAfterInterrupt = false;
  }
  else if((addr & 0x17) == 0x15)    // Write timer divide by 8
  {
    myTimer = value;
    myIntervalShift = 3;
    myCyclesWhenTimerSet = mySystem->cycles();
    myTimerReadAfterInterrupt = false;
  }
  else if((addr & 0x17) == 0x16)    // Write timer divide by 64
  {
    myTimer = value;
    myIntervalShift = 6;
    myCyclesWhenTimerSet = mySystem->cycles();
    myTimerReadAfterInterrupt = false;
  }
  else if((addr & 0x17) == 0x17)    // Write timer divide by 1024
  {
    myTimer = value;
    myIntervalShift = 10;
    myCyclesWhenTimerSet = mySystem->cycles();
    myTimerReadAfterInterrupt = false;
  }
  else if((addr & 0x14) == 0x04)    // Write Edge Detect Control
  {
#ifdef DEBUG_ACCESSES
    ale::Logger::Error << "M6532 Poke (Write Edge Detect): "
                       << ((addr & 0x02) ? "PA7 enabled" : "PA7 disabled")
                       << ", "
                       << ((addr & 0x01) ? "Positive edge" : "Negative edge")
                       << std::endl;
#endif
  }
  else
  {
#ifdef DEBUG_ACCESSES
    ale::Logger::Error << "BAD M6532 Poke: " << hex << addr << std::endl;
#endif
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool M6532::save(Serializer& out)
{
  std::string device = name();

  try
  {
    out.putString(device);

    // Output the RAM
    out.putInt(128);
    for(uint32_t t = 0; t < 128; ++t)
      out.putInt(myRAM[t]);

    out.putInt(myTimer);
    out.putInt(myIntervalShift);
    out.putInt(myCyclesWhenTimerSet);
    out.putInt(myCyclesWhenInterruptReset);
    out.putBool(myTimerReadAfterInterrupt);
    out.putInt(myDDRA);
    out.putInt(myDDRB);
  }
  catch(char *msg)
  {
    ale::Logger::Error << msg << std::endl;
    return false;
  }
  catch(...)
  {
    ale::Logger::Error << "Unknown error in save state for " << device << std::endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool M6532::load(Deserializer& in)
{
  std::string device = name();

  try
  {
    if(in.getString() != device)
      return false;

    // Input the RAM
    uint32_t limit = (uint32_t) in.getInt();
    for(uint32_t t = 0; t < limit; ++t)
      myRAM[t] = (uint8_t) in.getInt();

    myTimer = (uint32_t) in.getInt();
    myIntervalShift = (uint32_t) in.getInt();
    myCyclesWhenTimerSet = (uint32_t) in.getInt();
    myCyclesWhenInterruptReset = (uint32_t) in.getInt();
    myTimerReadAfterInterrupt = in.getBool();

    myDDRA = (uint8_t) in.getInt();
    myDDRB = (uint8_t) in.getInt();
  }
  catch(char *msg)
  {
    ale::Logger::Error << msg << std::endl;
    return false;
  }
  catch(...)
  {
    ale::Logger::Error << "Unknown error in load state for " << device << std::endl;
    return false;
  }

  return true;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6532::M6532(const M6532& c)
    : myConsole(c.myConsole)
{
  assert(false);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6532& M6532::operator = (const M6532&)
{
  assert(false);

  return *this;
}

}  // namespace stella
}  // namespace ale
