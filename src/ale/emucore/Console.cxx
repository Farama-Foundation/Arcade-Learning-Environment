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
// $Id: Console.cxx,v 1.128 2007/07/27 13:49:16 stephena Exp $
//============================================================================

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>

#include "ale/emucore/Cart.hxx"
#include "ale/emucore/Console.hxx"
#include "ale/emucore/Control.hxx"
#include "ale/emucore/Event.hxx"
#include "ale/emucore/Joystick.hxx"
#include "ale/emucore/M6502Hi.hxx"
#include "ale/emucore/M6502Low.hxx"
#include "ale/emucore/M6532.hxx"
#include "ale/emucore/MediaSrc.hxx"
#include "ale/emucore/Paddles.hxx"
#include "ale/emucore/Props.hxx"
#include "ale/emucore/PropsSet.hxx"
#include "ale/emucore/Settings.hxx"
#include "ale/emucore/Sound.hxx"
#include "ale/emucore/Switches.hxx"
#include "ale/emucore/System.hxx"
#include "ale/emucore/TIA.hxx"
#include "ale/emucore/OSystem.hxx"

#include "ale/common/Log.hpp"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Console::Console(OSystem* osystem, Cartridge* cart, const Properties& props)
  : myOSystem(osystem),
    myProperties(props)
{
  myControllers[0] = 0;
  myControllers[1] = 0;
  myMediaSource = 0;
  mySwitches = 0;
  mySystem = 0;
  myEvent = 0;

  myEvent = myOSystem->event();

  // Setup the controllers based on properties
  const std::string& left  = myProperties.get(Controller_Left);
  const std::string& right = myProperties.get(Controller_Right);

  // Swap the ports if necessary
  int leftPort, rightPort;
  if(myProperties.get(Console_SwapPorts) == "NO")
  {
    leftPort = 0; rightPort = 1;
  }
  else
  {
    leftPort = 1; rightPort = 0;
  }

  // Also check if we should swap the paddles plugged into a jack
  bool swapPaddles = myProperties.get(Controller_SwapPaddles) == "YES";

  // Construct left controller
  if(left == "PADDLES")
  {
    myControllers[leftPort] = new Paddles(Controller::Left, *myEvent, swapPaddles);
  }
  else
  {
    myControllers[leftPort] = new Joystick(Controller::Left, *myEvent);
  }

  // Construct right controller
  if(right == "PADDLES")
  {
    myControllers[rightPort] = new Paddles(Controller::Right, *myEvent, swapPaddles);
  }
  else
  {
    myControllers[rightPort] = new Joystick(Controller::Right, *myEvent);
  }

  // Create switches for the console
  mySwitches = new Switches(*myEvent, myProperties);

  // Now, we can construct the system and components
  mySystem = new System(myOSystem->settings());

  // Inform the controllers about the system
  myControllers[0]->setSystem(mySystem);
  myControllers[1]->setSystem(mySystem);

  M6502* m6502;
  if(myOSystem->settings().getString("cpu") == "low") {
    m6502 = new M6502Low(1);
  }
  else {
    m6502 = new M6502High(1);
  }

  M6532* m6532 = new M6532(*this);

  TIA *tia = new TIA(*this, myOSystem->settings());
  tia->setSound(myOSystem->sound());

  mySystem->attach(m6502);
  mySystem->attach(m6532);
  mySystem->attach(tia);
  mySystem->attach(cart);

  // Remember what my media source is
  myMediaSource = tia;

  // Query some info about this console
  std::ostringstream buf;
  buf << "  Cart Name: " << myProperties.get(Cartridge_Name) << std::endl
      << "  Cart MD5:  " << myProperties.get(Cartridge_MD5) << std::endl;

  // Auto-detect NTSC/PAL mode if it's requested
  myDisplayFormat = myProperties.get(Display_Format);
  buf << "  Display Format:  " << myDisplayFormat;
  if(myDisplayFormat == "AUTO-DETECT" ||
     myOSystem->settings().getBool("rominfo"))
  {
    // Run the system for 60 frames, looking for PAL scanline patterns
    // We assume the first 30 frames are garbage, and only consider
    // the second 30 (useful to get past SuperCharger BIOS)
    // Unfortunately, this means we have to always enable 'fastscbios',
    // since otherwise the BIOS loading will take over 250 frames!
    mySystem->reset();
    int palCount = 0;
    for(int i = 0; i < 60; ++i)
    {
      myMediaSource->update();
      if(i >= 30 && myMediaSource->scanlines() > 285)
        ++palCount;
    }

    myDisplayFormat = (palCount >= 15) ? "PAL" : "NTSC";
    if(myProperties.get(Display_Format) == "AUTO-DETECT")
      buf << " ==> " << myDisplayFormat;
  }
  buf << std::endl << cart->about();

  // Make sure height is set properly for PAL ROM
  if((myDisplayFormat == "PAL" || myDisplayFormat == "SECAM") &&
      myProperties.get(Display_Height) == "210")
    myProperties.set(Display_Height, "250");

  // Reset, the system to its power-on state
  mySystem->reset();

  myAboutString = buf.str();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Console::~Console()
{
  delete mySystem;
  delete mySwitches;
  delete myControllers[0];
  delete myControllers[1];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::setProperties(const Properties& props)
{
  myProperties = props;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint32_t Console::getFrameRate() const
{
  // Set the correct framerate based on the format of the ROM
  // This can be overridden by changing the framerate in the
  // VideoDialog box or on the commandline, but it can't be saved
  // (ie, framerate is now solely determined based on ROM format).
  int framerate = myOSystem->settings().getInt("framerate");
  if(framerate == -1)
  {
    if(myDisplayFormat == "NTSC" || myDisplayFormat == "PAL60")
      framerate = 60;
    else if(myDisplayFormat == "PAL" || myDisplayFormat == "SECAM")
      framerate = 50;
    else
      framerate = 60;
  }

  return framerate;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Console::Console(const Console& console)
  : myOSystem(console.myOSystem)
{
  // TODO: Write this method
  assert(false);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Console& Console::operator = (const Console&)
{
  // TODO: Write this method
  assert(false);

  return *this;
}

}  // namespace stella
}  // namespace ale
