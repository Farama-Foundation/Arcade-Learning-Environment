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
// $Id: Console.hxx,v 1.61 2007/07/27 13:49:16 stephena Exp $
//============================================================================

#ifndef CONSOLE_HXX
#define CONSOLE_HXX

namespace ale {
namespace stella {

class OSystem;
class Console;
class Controller;
class Event;
class MediaSource;
class Switches;
class System;

}  // namespace stella
}  // namespace ale

#include "ale/emucore/Control.hxx"
#include "ale/emucore/Props.hxx"
#include "ale/emucore/TIA.hxx"
#include "ale/emucore/Cart.hxx"

namespace ale {
namespace stella {

/**
  This class represents the entire game console.

  @author  Bradford W. Mott
  @version $Id: Console.hxx,v 1.61 2007/07/27 13:49:16 stephena Exp $
*/
class Console
{
  public:
    /**
      Create a new console for emulating the specified game using the
      given game image and operating system.

      @param osystem  The OSystem object to use
      @param cart     The cartridge to use with this console
      @param props    The properties for the cartridge
    */
    Console(OSystem* osystem, Cartridge* cart, const Properties& props);

    /**
      Create a new console object by copying another one

      @param console The object to copy
    */
    Console(const Console& console);

    /**
      Destructor
    */
    virtual ~Console();

  public:
    /**
      Get the controller plugged into the specified jack

      @return The specified controller
    */
    Controller& controller(Controller::Jack jack) const
    {
      return (jack == Controller::Left) ? *myControllers[0] : *myControllers[1];
    }

    /**
      Get the MediaSource for this console

      @return The mediasource
    */
    MediaSource& mediaSource() const { return *myMediaSource; }

    /**
      Get the properties being used by the game

      @return The properties being used by the game
    */
    const Properties& properties() const { return myProperties; }

    /**
      Get the console switches

      @return The console switches
    */
    Switches& switches() const { return *mySwitches; }

    /**
      Get the 6502 based system used by the console to emulate the game

      @return The 6502 based system
    */
    System& system() const { return *mySystem; }

    /**
      Returns the OSystem for this emulator.

      @return The OSystem.
    */
    OSystem& osystem() const { return *myOSystem; }

    /**
      Set the properties to those given

      @param The properties to use for the current game
    */
    void setProperties(const Properties& props);

    /**
      Query some information about this console.
    */
    const std::string& about() const { return myAboutString; }

  public:
    /**
      Overloaded assignment operator

      @param console The console object to set myself equal to
      @return Myself after assignment has taken place
    */
    Console& operator = (const Console& console);

  public:
    /**
      Query the currently selected display format (NTSC/PAL/PAL60).
    */
    std::string getFormat() const { return myDisplayFormat; }

    /**
      Returns the framerate based on a number of factors
      (whether 'framerate' is set, what display format is in use, etc)
    */
    uint32_t getFrameRate() const;

  private:
    // Pointer to the osystem object
    OSystem* myOSystem;

    // Pointers to the left and right controllers
    Controller* myControllers[2];

    // Pointer to the event object to use
    Event* myEvent;

    // Pointer to the media source object
    MediaSource* myMediaSource;

    // Properties for the game
    Properties myProperties;

    // Pointer to the switches on the front of the console
    Switches* mySwitches;

    // Pointer to the 6502 based system being emulated
    System* mySystem;

    // The currently defined display format (NTSC/PAL/PAL60)
    std::string myDisplayFormat;

    // Contains info about this console in string format
    std::string myAboutString;

};

}  // namespace stella
}  // namespace ale

#endif
