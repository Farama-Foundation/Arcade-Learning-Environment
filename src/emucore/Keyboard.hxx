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
// $Id: Keyboard.hxx,v 1.6 2007/01/01 18:04:48 stephena Exp $
//============================================================================

#ifndef KEYBOARD_HXX
#define KEYBOARD_HXX

#include "m6502/src/bspf/src/bspf.hxx"
#include "Control.hxx"

/**
  The standard Atari 2600 keyboard controller

  @author  Bradford W. Mott
  @version $Id: Keyboard.hxx,v 1.6 2007/01/01 18:04:48 stephena Exp $
*/
class Keyboard : public Controller
{
  public:
    /**
      Create a new keyboard controller plugged into the specified jack

      @param jack The jack the controller is plugged into
      @param event The event object to use for events
    */
    Keyboard(Jack jack, const Event& event);

    /**
      Destructor
    */
    virtual ~Keyboard();

  public:
    /**
      Read the value of the specified digital pin for this controller.

      @param pin The pin of the controller jack to read
      @return The state of the pin
    */
    virtual bool read(DigitalPin pin);

    /**
      Read the resistance at the specified analog pin for this controller.
      The returned value is the resistance measured in ohms.

      @param pin The pin of the controller jack to read
      @return The resistance at the specified pin
    */
    virtual Int32 read(AnalogPin pin);

    /**
      Write the given value to the specified digital pin for this
      controller.  Writing is only allowed to the pins associated
      with the PIA.  Therefore you cannot write to pin six.

      @param pin The pin of the controller jack to write to
      @param value The value to write to the pin
    */
    virtual void write(DigitalPin pin, bool value);

  private:
    // State of the output pins
    uInt8 myPinState;
};

#endif
