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
// $Id: Switches.cxx,v 1.7 2007/01/01 18:04:50 stephena Exp $
//============================================================================

#include "Event.hxx"
#include "Props.hxx"
#include "Switches.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Switches::Switches(const Event& event, const Properties& properties)
    : myEvent(event),
      mySwitches(0xFF)
{
  if(properties.get(Console_RightDifficulty) == "B")
  {
    mySwitches &= ~0x80;
  }
  else
  {
    mySwitches |= 0x80;
  }

  if(properties.get(Console_LeftDifficulty) == "B")
  {
    mySwitches &= ~0x40;
  }
  else
  {
    mySwitches |= 0x40;
  }

  if(properties.get(Console_TelevisionType) == "COLOR")
  {
    mySwitches |= 0x08;
  }
  else
  {
    mySwitches &= ~0x08;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Switches::~Switches()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
uInt8 Switches::read()
{
  if(myEvent.get(Event::ConsoleColor) != 0)
  {
    mySwitches |= 0x08;
  }
  else if(myEvent.get(Event::ConsoleBlackWhite) != 0)
  {
    mySwitches &= ~0x08;
  }

  if(myEvent.get(Event::ConsoleRightDifficultyA) != 0)
  {
    mySwitches |= 0x80;
  }
  else if(myEvent.get(Event::ConsoleRightDifficultyB) != 0) 
  {
    mySwitches &= ~0x80;
  }

  if(myEvent.get(Event::ConsoleLeftDifficultyA) != 0)
  {
    mySwitches |= 0x40;
  }
  else if(myEvent.get(Event::ConsoleLeftDifficultyB) != 0)
  {
    mySwitches &= ~0x40;
  }

  if(myEvent.get(Event::ConsoleSelect) != 0)
  {
    mySwitches &= ~0x02;
  }
  else 
  {
    mySwitches |= 0x02;
  }

  if(myEvent.get(Event::ConsoleReset) != 0)
  {
    mySwitches &= ~0x01;
  }
  else 
  {
    mySwitches |= 0x01;
  }

  return mySwitches;
}

