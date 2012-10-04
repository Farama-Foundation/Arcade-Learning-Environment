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
// $Id: Paddles.cxx,v 1.8 2007/01/05 17:54:23 stephena Exp $
//============================================================================

#include "Event.hxx"
#include "Paddles.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Paddles::Paddles(Jack jack, const Event& event, bool swap)
  : Controller(jack, event, Controller::Paddles)
{
  // Swap the paddle events, from paddle 0 <=> 1 and paddle 2 <=> 3
  if(!swap)
  {
    // Pin Three
    myPinEvents[0][0] = Event::PaddleOneFire;
    myPinEvents[0][1] = Event::PaddleThreeFire;

    // Pin Four
    myPinEvents[1][0] = Event::PaddleZeroFire;
    myPinEvents[1][1] = Event::PaddleTwoFire;

    // Pin Five
    myPinEvents[2][0] = Event::PaddleOneResistance;
    myPinEvents[2][1] = Event::PaddleThreeResistance;

    // Pin Nine
    myPinEvents[3][0] = Event::PaddleZeroResistance;
    myPinEvents[3][1] = Event::PaddleTwoResistance;
  }
  else
  {
    // Pin Three (swapped)
    myPinEvents[0][0] = Event::PaddleZeroFire;
    myPinEvents[0][1] = Event::PaddleTwoFire;

    // Pin Four (swapped)
    myPinEvents[1][0] = Event::PaddleOneFire;
    myPinEvents[1][1] = Event::PaddleThreeFire;

    // Pin Five (swapped)
    myPinEvents[2][0] = Event::PaddleZeroResistance;
    myPinEvents[2][1] = Event::PaddleTwoResistance;

    // Pin Nine (swapped)
    myPinEvents[3][0] = Event::PaddleOneResistance;
    myPinEvents[3][1] = Event::PaddleThreeResistance;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Paddles::~Paddles()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Paddles::read(DigitalPin pin)
{
  switch(pin)
  {
    case Three:
      return (myJack == Left) ? (myEvent.get(myPinEvents[0][0]) == 0) : 
          (myEvent.get(myPinEvents[0][1]) == 0);

    case Four:
      return (myJack == Left) ? (myEvent.get(myPinEvents[1][0]) == 0) : 
          (myEvent.get(myPinEvents[1][1]) == 0);

    default:
      // Other pins are not connected (floating high)
      return true;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Int32 Paddles::read(AnalogPin pin)
{
  switch(pin)
  {
    case Five:
      return (myJack == Left) ? myEvent.get(myPinEvents[2][0]) : 
          myEvent.get(myPinEvents[2][1]);

    case Nine:
      return (myJack == Left) ? myEvent.get(myPinEvents[3][0]) : 
          myEvent.get(myPinEvents[3][1]);

    default:
      return maximumResistance;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Paddles::write(DigitalPin, bool)
{
  // Writing doesn't do anything to the paddles...
}
