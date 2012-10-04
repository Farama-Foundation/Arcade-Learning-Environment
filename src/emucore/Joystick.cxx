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
// $Id: Joystick.cxx,v 1.7 2007/01/05 17:54:23 stephena Exp $
//============================================================================

#include <assert.h>
#include "Event.hxx"
#include "Joystick.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Joystick::Joystick(Jack jack, const Event& event)
  : Controller(jack, event, Controller::Joystick)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Joystick::~Joystick()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Joystick::read(DigitalPin pin)
{
  switch(pin)
  {
    case One:
      return (myJack == Left) ? (myEvent.get(Event::JoystickZeroUp) == 0) : 
          (myEvent.get(Event::JoystickOneUp) == 0);

    case Two:
      return (myJack == Left) ? (myEvent.get(Event::JoystickZeroDown) == 0) : 
          (myEvent.get(Event::JoystickOneDown) == 0);

    case Three:
      return (myJack == Left) ? (myEvent.get(Event::JoystickZeroLeft) == 0) : 
          (myEvent.get(Event::JoystickOneLeft) == 0);

    case Four:
      return (myJack == Left) ? (myEvent.get(Event::JoystickZeroRight) == 0) : 
          (myEvent.get(Event::JoystickOneRight) == 0);

    case Six:
      return (myJack == Left) ? (myEvent.get(Event::JoystickZeroFire) == 0) : 
          (myEvent.get(Event::JoystickOneFire) == 0);

    default:
      return true;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Int32 Joystick::read(AnalogPin)
{
  // Analog pins are not connect in joystick so we have infinite resistance 
  return maximumResistance;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Joystick::write(DigitalPin, bool)
{
  // Writing doesn't do anything to the joystick...
}
