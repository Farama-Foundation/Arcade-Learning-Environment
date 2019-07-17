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

#include "AtariVox.hxx"
#include "Booster.hxx"
#include "Cart.hxx"
#include "Console.hxx"
#include "Control.hxx"
#include "Driving.hxx"
#include "Event.hxx"
//ALE #include "EventHandler.hxx"
#include "Joystick.hxx"
#include "Keyboard.hxx"
#include "M6502Hi.hxx"
#include "M6502Low.hxx"
#include "M6532.hxx"
#include "MediaSrc.hxx"
#include "Paddles.hxx"
#include "Props.hxx"
#include "PropsSet.hxx"
#include "Settings.hxx" 
#include "Sound.hxx"
#include "Switches.hxx"
#include "System.hxx"
#include "TIA.hxx"
//ALE #include "FrameBuffer.hxx"
#include "OSystem.hxx"
//ALE #include "Menu.hxx"
//ALE #include "CommandMenu.hxx"
#include "Version.hxx"
#ifdef DEBUGGER_SUPPORT
  #include "Debugger.hxx"
#endif

#ifdef CHEATCODE_SUPPORT
  #include "CheatManager.hxx"
#endif
using namespace std;
#include "../common/Log.hpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Console::Console(OSystem* osystem, Cartridge* cart, const Properties& props)
  : myOSystem(osystem),
    myProperties(props),
    myUserPaletteDefined(false)
{
  myControllers[0] = 0;
  myControllers[1] = 0;
  myMediaSource = 0;
  mySwitches = 0;
  mySystem = 0;
  myEvent = 0;
  
  // Attach the event subsystem to the current console
  //ALE  myEvent = myOSystem->eventHandler().event();
  myEvent = myOSystem->event();

  // Setup the controllers based on properties
  const string& left  = myProperties.get(Controller_Left);
  const string& right = myProperties.get(Controller_Right);

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
  if(left == "BOOSTER-GRIP")
  {
    myControllers[leftPort] = new BoosterGrip(Controller::Left, *myEvent);
  }
  else if(left == "DRIVING")
  {
    myControllers[leftPort] = new Driving(Controller::Left, *myEvent);
  }
  else if((left == "KEYBOARD") || (left == "KEYPAD"))
  {
    myControllers[leftPort] = new Keyboard(Controller::Left, *myEvent);
  }
  else if(left == "PADDLES")
  {
    myControllers[leftPort] = new Paddles(Controller::Left, *myEvent, swapPaddles);
  }
  else
  {
    myControllers[leftPort] = new Joystick(Controller::Left, *myEvent);
  }
 
  // Construct right controller
  if(right == "BOOSTER-GRIP")
  {
    myControllers[rightPort] = new BoosterGrip(Controller::Right, *myEvent);
  }
  else if(right == "DRIVING")
  {
    myControllers[rightPort] = new Driving(Controller::Right, *myEvent);
  }
  else if((right == "KEYBOARD") || (right == "KEYPAD"))
  {
    myControllers[rightPort] = new Keyboard(Controller::Right, *myEvent);
  }
  else if(right == "PADDLES")
  {
    myControllers[rightPort] = new Paddles(Controller::Right, *myEvent, swapPaddles);
  }
#ifdef ATARIVOX_SUPPORT
  else if(right == "ATARIVOX")
  {
    myControllers[rightPort] = new AtariVox(Controller::Right, *myEvent);
  }
#endif
  else
  {
    myControllers[rightPort] = new Joystick(Controller::Right, *myEvent);
  }

  // Create switches for the console
  mySwitches = new Switches(*myEvent, myProperties);

  // Now, we can construct the system and components
  mySystem = new System();

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
#ifdef DEBUGGER_SUPPORT
  m6502->attach(myOSystem->debugger());
#endif

  M6532* m6532 = new M6532(*this);

  TIA *tia = new TIA(*this, myOSystem->settings());
  tia->setSound(myOSystem->sound());

  mySystem->attach(m6502);
  mySystem->attach(m6532);
  mySystem->attach(tia);
  mySystem->attach(cart);

  // Remember what my media source is
  myMediaSource = tia;
  myCart = cart;
  myRiot = m6532;

  // Query some info about this console
  ostringstream buf;
  buf << "  Cart Name: " << myProperties.get(Cartridge_Name) << endl
      << "  Cart MD5:  " << myProperties.get(Cartridge_MD5) << endl;

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
  buf << endl << cart->about();

  // Make sure height is set properly for PAL ROM
  if((myDisplayFormat == "PAL" || myDisplayFormat == "SECAM") &&
      myProperties.get(Display_Height) == "210")
    myProperties.set(Display_Height, "250");

  // Reset, the system to its power-on state
  mySystem->reset();

  // Bumper Bash requires all 4 directions
//ALE   const string& md5 = myProperties.get(Cartridge_MD5);
//ALE   bool allow = (md5 == "aa1c41f86ec44c0a44eb64c332ce08af" ||
//ALE                 md5 == "1bf503c724001b09be79c515ecfcbd03");
//ALE  myOSystem->eventHandler().allowAllDirections(allow);

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
void Console::toggleFormat()
{
  int framerate = 60;
  if(myDisplayFormat == "NTSC")
  {
    myDisplayFormat = "PAL";
    myProperties.set(Display_Format, myDisplayFormat);
    mySystem->reset();
    //ALE  myOSystem->frameBuffer().showMessage("PAL Mode");
    framerate = 50;
  }
  else if(myDisplayFormat == "PAL")
  {
    myDisplayFormat = "PAL60";
    myProperties.set(Display_Format, myDisplayFormat);
    mySystem->reset();
    //ALE  myOSystem->frameBuffer().showMessage("PAL60 Mode");
    framerate = 60;
  }
  else if(myDisplayFormat == "PAL60")
  {
    myDisplayFormat = "SECAM";
    myProperties.set(Display_Format, myDisplayFormat);
    mySystem->reset();
    //ALE  myOSystem->frameBuffer().showMessage("SECAM Mode");
    framerate = 50;
  }
  else if(myDisplayFormat == "SECAM")
  {
    myDisplayFormat = "NTSC";
    myProperties.set(Display_Format, myDisplayFormat);
    mySystem->reset();
    //ALE  myOSystem->frameBuffer().showMessage("NTSC Mode");
    framerate = 60;
  }

  myOSystem->colourPalette().setPalette(myOSystem->settings().getString("palette"), myDisplayFormat);
  myOSystem->setFramerate(framerate);
  myOSystem->sound().setFrameRate(framerate);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::togglePalette()
{
  string palette, message;
  palette = myOSystem->settings().getString("palette");

  if(palette == "standard")       // switch to z26
  {
    palette = "z26";
    message = "Z26 palette";
  }
  else if(palette == "z26")       // switch to user or standard
  {
    // If we have a user-defined palette, it will come next in
    // the sequence; otherwise loop back to the standard one
    if(myUserPaletteDefined)
    {
      palette = "user";
      message = "User-defined palette";
    }
    else
    {
      palette = "standard";
      message = "Standard Stella palette";
    }
  }
  else if(palette == "user")  // switch to standard
  {
    palette = "standard";
    message = "Standard Stella palette";
  }
  else  // switch to standard mode if we get this far
  {
    palette = "standard";
    message = "Standard Stella palette";
  }

  myOSystem->settings().setString("palette", palette);
  //ALE  myOSystem->frameBuffer().showMessage(message);

  myOSystem->colourPalette().setPalette(palette, myDisplayFormat);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::togglePhosphor()
{
  // MGB: This method is deprecated. 
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::setProperties(const Properties& props)
{
  myProperties = props;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::initializeVideo(bool full)
{
  if(full)
  {
    string title = string("Stella ") + STELLA_VERSION +
                   ": \"" + myProperties.get(Cartridge_Name) + "\"";
    // ALE myOSystem->frameBuffer().initialize(title,
    // ALE                                     myMediaSource->width() << 1,
    // ALE                                     myMediaSource->height());
  }

  //ALE   bool enable = myProperties.get(Display_Phosphor) == "YES";
  //ALE   int blend = atoi(myProperties.get(Display_PPBlend).c_str());
  //ALE  myOSystem->frameBuffer().enablePhosphor(enable, blend);
  myOSystem->colourPalette().setPalette(myOSystem->settings().getString("palette"), myDisplayFormat);

  myOSystem->setFramerate(getFrameRate());
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::initializeAudio()
{
  // Initialize the sound interface.
  // The # of channels can be overridden in the AudioDialog box or on
  // the commandline, but it can't be saved.
  const string& sound = myProperties.get(Cartridge_Sound);
  uInt32 channels = (sound == "STEREO" ? 2 : 1);

  myOSystem->sound().close();
  myOSystem->sound().setChannels(channels);
  myOSystem->sound().setFrameRate(getFrameRate());
  myOSystem->sound().initialize();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/* Original frying research and code by Fred Quimby.
   I've tried the following variations on this code:
   - Both OR and Exclusive OR instead of AND. This generally crashes the game
     without ever giving us realistic "fried" effects.
   - Loop only over the RIOT RAM. This still gave us frying-like effects, but
     it seemed harder to duplicate most effects. I have no idea why, but
     munging the TIA regs seems to have some effect (I'd think it wouldn't).

   Fred says he also tried mangling the PC and registers, but usually it'd just
   crash the game (e.g. black screen, no way out of it).

   It's definitely easier to get some effects (e.g. 255 lives in Battlezone)
   with this code than it is on a real console. My guess is that most "good"
   frying effects come from a RIOT location getting cleared to 0. Fred's
   code is more likely to accomplish this than frying a real console is...

   Until someone comes up with a more accurate way to emulate frying, I'm
   leaving this as Fred posted it.   -- B.
*/
void Console::fry() const
{
  for (int ZPmem=0; ZPmem<0x100; ZPmem += myOSystem->rng().next() % 4)
    mySystem->poke(ZPmem, mySystem->peek(ZPmem) & (uInt8)myOSystem->rng().next() % 256);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::changeYStart(int direction)
{
  Int32 ystart = atoi(myProperties.get(Display_YStart).c_str());
  ostringstream strval;
  string message;

  if(direction == +1)    // increase YStart
  {
    ystart++;
    if(ystart > 64)
    {
      //ALE   myOSystem->frameBuffer().showMessage("YStart at maximum");
      return;
    }
  }
  else if(direction == -1)  // decrease YStart
  {
    ystart--;
    if(ystart < 0)
    {
      //ALE   myOSystem->frameBuffer().showMessage("YStart at minimum");
      return;
    }
  }
  else
    return;

  strval << ystart;
  myProperties.set(Display_YStart, strval.str());
  ((TIA*)myMediaSource)->frameReset();
  //ALE  myOSystem->frameBuffer().refresh();

  message = "YStart ";
  message += strval.str();
  //ALE  myOSystem->frameBuffer().showMessage(message);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::changeHeight(int direction)
{
  Int32 height = atoi(myProperties.get(Display_Height).c_str());
  ostringstream strval;
  string message;

  if(direction == +1)    // increase Height
  {
    height++;
    if(height > 256)
    {
      //ALE  myOSystem->frameBuffer().showMessage("Height at maximum");
      return;
    }
  }
  else if(direction == -1)  // decrease Height
  {
    height--;
    if(height < 200)
    {
      //ALE  myOSystem->frameBuffer().showMessage("Height at minimum");
      return;
    }
  }
  else
    return;

  strval << height;
  myProperties.set(Display_Height, strval.str());
  ((TIA*)myMediaSource)->frameReset();
  initializeVideo();  // takes care of refreshing the screen

  message = "Height ";
  message += strval.str();
  //ALE  myOSystem->frameBuffer().showMessage(message);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::toggleTIABit(TIA::TIABit bit, const string& bitname, bool show) const
{
  bool result = ((TIA*)myMediaSource)->toggleBit(bit);
  string message = bitname + (result ? " enabled" : " disabled");
  //ALE  myOSystem->frameBuffer().showMessage(message);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Console::enableBits(bool enable) const
{
  ((TIA*)myMediaSource)->enableBits(enable);
  string message = string("TIA bits") + (enable ? " enabled" : " disabled");
  //ALE  myOSystem->frameBuffer().showMessage(message);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 Console::getFrameRate() const
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
