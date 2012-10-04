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
// $Id: Control.cxx,v 1.6 2007/01/05 17:54:09 stephena Exp $
//============================================================================

#include <assert.h>
#include "Control.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Controller::Controller(Jack jack, const Event& event, Type type)
  : myJack(jack),
    myEvent(event),
    myType(type)
{
}
 
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Controller::~Controller()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const Controller::Type Controller::type()
{
  return myType;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const Int32 Controller::maximumResistance = 0x7FFFFFFF;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const Int32 Controller::minimumResistance = 0x00000000;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Controller::Controller(const Controller& c)
  : myJack(c.myJack),
    myEvent(c.myEvent),
    myType(c.myType)
{
  assert(false);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Controller& Controller::operator = (const Controller&)
{
  assert(false);
  return *this;
}
