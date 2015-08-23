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

class Console;
class Controller;
class Event;
class MediaSource;
class Switches;
class System;

#include "m6502/src/bspf/src/bspf.hxx"
#include "Control.hxx"
#include "Props.hxx"
#include "TIA.hxx"
#include "Cart.hxx"
#include "M6532.hxx"
#include "AtariVox.hxx"

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
      Get the cartridge used by the console which contains the ROM code

      @return The cartridge for this console
    */
    Cartridge& cartridge() const { return *myCart; }

    /**
      Get the 6532 used by the console

      @return The 6532 for this console
    */
    M6532& riot() const { return *myRiot; }

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
      Toggle between NTSC/PAL/PAL60 display format.
    */
    void toggleFormat();

    /**
      Query the currently selected display format (NTSC/PAL/PAL60).
    */
    std::string getFormat() const { return myDisplayFormat; }

    /**
      Toggle between the available palettes.
    */
    void togglePalette();

    /**
      Toggles phosphor effect.
    */
    void togglePhosphor();

    /**
      Initialize the video subsystem wrt this class.
      This is required for changing window size, title, etc.

      @param full  Whether we want a full initialization,
                   or only reset certain attributes.
    */
    void initializeVideo(bool full = true);

    /**
      Initialize the audio subsystem wrt this class.
      This is required any time the sound settings change.
    */
    void initializeAudio();

    /**
      "Fry" the Atari (mangle memory/TIA contents)
    */
    void fry() const;

    /**
      Change the "Display.YStart" variable.

      @param direction +1 indicates increase, -1 indicates decrease.
    */
    void changeYStart(int direction);

    /**
      Change the "Display.Height" variable.

      @param direction +1 indicates increase, -1 indicates decrease.
    */
    void changeHeight(int direction);

    /**
      Toggles the TIA bit specified in the method name.
    */
    void toggleP0Bit() const { toggleTIABit(TIA::P0, "P0"); }
    void toggleP1Bit() const { toggleTIABit(TIA::P1, "P1"); }
    void toggleM0Bit() const { toggleTIABit(TIA::M0, "M0"); }
    void toggleM1Bit() const { toggleTIABit(TIA::M1, "M1"); }
    void toggleBLBit() const { toggleTIABit(TIA::BL, "BL"); }
    void togglePFBit() const { toggleTIABit(TIA::PF, "PF"); }
    void enableBits(bool enable) const;

#ifdef ATARIVOX_SUPPORT
    AtariVox *atariVox() { return vox; }
#endif

  private:
    void toggleTIABit(TIA::TIABit bit, const std::string& bitname, bool show = true) const;

    /**
      Returns the framerate based on a number of factors
      (whether 'framerate' is set, what display format is in use, etc)
    */
    uInt32 getFrameRate() const;

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

    // Pointer to the Cartridge (the debugger needs it)
    Cartridge *myCart;

    // Pointer to the 6532 (aka RIOT) (the debugger needs it)
    // A RIOT of my own! (...with apologies to The Clash...)
    M6532 *myRiot;

    //Random number generator
    mutable Random randNumGen;

#ifdef ATARIVOX_SUPPORT
    AtariVox *vox;
#endif

    // The currently defined display format (NTSC/PAL/PAL60)
    std::string myDisplayFormat;

    // Indicates whether an external palette was found and
    // successfully loaded
    bool myUserPaletteDefined;

    // Contains info about this console in string format
    std::string myAboutString;

};

#endif
