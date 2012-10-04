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
// $Id: Driving.cxx,v 1.11 2007/02/22 02:15:46 stephena Exp $
//============================================================================

#include <cassert>

#include "Event.hxx"
#include "Driving.hxx"
#include "System.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Driving::Driving(Jack jack, const Event& event)
  : Controller(jack, event, Controller::Driving),
    myCounter(0)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Driving::~Driving()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Driving::read(DigitalPin pin)
{
  // Gray codes for clockwise rotation
  static const uInt8 clockwise[] = { 0x03, 0x01, 0x00, 0x02 };

  // Gray codes for counter-clockwise rotation
  static const uInt8 counterclockwise[] = { 0x03, 0x02, 0x00, 0x01 };

  // Delay used for moving through the gray code tables
    const uInt32 delay = 20;

  switch(pin)
  {
    case One:
      ++myCounter;

      if(myJack == Left)
      {
        if(myEvent.get(Event::DrivingZeroCounterClockwise) != 0)
        {
          return (counterclockwise[(myCounter / delay) & 0x03] & 0x01) != 0;
        }
        else if(myEvent.get(Event::DrivingZeroClockwise) != 0)
        {
          return (clockwise[(myCounter / delay) & 0x03] & 0x01) != 0;
        }
        else 
          return(myEvent.get(Event::DrivingZeroValue) & 0x01);
      }
      else
      {
        if(myEvent.get(Event::DrivingOneCounterClockwise) != 0)
        {
          return (counterclockwise[(myCounter / delay) & 0x03] & 0x01) != 0;
        }
        else if(myEvent.get(Event::DrivingOneClockwise) != 0)
        {
          return (clockwise[(myCounter / delay) & 0x03] & 0x01) != 0;
        }
        else 
          return(myEvent.get(Event::DrivingOneValue) & 0x01);
      }

    case Two:
      if(myJack == Left)
      {
        if(myEvent.get(Event::DrivingZeroCounterClockwise) != 0)
        {
          return (counterclockwise[(myCounter / delay) & 0x03] & 0x02) != 0;
        }
        else if(myEvent.get(Event::DrivingZeroClockwise) != 0)
        {
          return (clockwise[(myCounter / delay) & 0x03] & 0x02) != 0;
        }
        else 
          return(myEvent.get(Event::DrivingZeroValue) & 0x02);
      }
      else
      {
        if(myEvent.get(Event::DrivingOneCounterClockwise) != 0)
        {
          return (counterclockwise[(myCounter / delay) & 0x03] & 0x02) != 0;
        }
        else if(myEvent.get(Event::DrivingOneClockwise) != 0)
        {
          return (clockwise[(myCounter / delay) & 0x03] & 0x02) != 0;
        }
        else 
          return(myEvent.get(Event::DrivingOneValue) & 0x02);
      }

    case Three:
      return true;

    case Four:
      return true;

    case Six:
      return (myJack == Left) ? (myEvent.get(Event::DrivingZeroFire) == 0) : 
          (myEvent.get(Event::DrivingOneFire) == 0);

    default:
      return true;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Int32 Driving::read(AnalogPin)
{
  // Analog pins are not connect in driving controller so we have 
  // infinite resistance 
  return maximumResistance;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Driving::write(DigitalPin, bool)
{
  // Writing doesn't do anything to the driving controller...
}
