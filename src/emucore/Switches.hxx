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
// $Id: Switches.hxx,v 1.4 2007/01/01 18:04:50 stephena Exp $
//============================================================================

#ifndef SWITCHES_HXX
#define SWITCHES_HXX

class Event;
class Properties;
class Switches;

#include "m6502/src/bspf/src/bspf.hxx"

/**
  This class represents the console switches of the game console.

  @author  Bradford W. Mott
  @version $Id: Switches.hxx,v 1.4 2007/01/01 18:04:50 stephena Exp $
*/
class Switches
{
  public:
    /**
      Create a new set of switches using the specified events and
      properties

      @param event The event object to use for events
    */
    Switches(const Event& event, const Properties& properties);
 
    /**
      Destructor
    */
    virtual ~Switches();

  public:
    /**
      Get the value of the console switches

      @return The 8 bits which represent the state of the console switches
    */
    uInt8 read();

  private:
    // Reference to the event object to use
    const Event& myEvent;

    // State of the console switches
    uInt8 mySwitches;
};
#endif

