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
// $Id: OSystem.cxx,v 1.108 2007/08/17 16:12:50 stephena Exp $
//============================================================================

#include <cassert>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <zlib.h>

#include "emucore/MD5.hxx"
#include "emucore/Settings.hxx"
#include "emucore/PropsSet.hxx"
#include "emucore/Event.hxx"
#include "emucore/OSystem.hxx"
#include "common/SoundSDL.hxx"

#define MAX_ROM_SIZE  512 * 1024

#include <time.h>


namespace fs = std::filesystem;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
OSystem::OSystem()
  : 
    myEvent(NULL),
    mySound(NULL),
    mySettings(NULL),
    myPropSet(NULL),
    myConsole(NULL),
    myRomFile(""),
    p_display_screen(NULL)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
OSystem::~OSystem()
{
  // Remove any game console that is currently attached
  deleteConsole();

  // OSystem takes responsibility for framebuffer and sound,
  // since it created them
  if (mySound != NULL)
    delete mySound;

  if (myPropSet != NULL)
    delete myPropSet;
  if (myEvent != NULL)
    delete myEvent; 
  if (p_display_screen != NULL) {
      delete p_display_screen;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool OSystem::create()
{
  // Delete the previous event object (if any).
  delete myEvent;

  // Create the event object which will be used for this handler
  myEvent = new Event();

  // Delete the previous properties set (if any).
  delete myPropSet;

  // Create a properties set for us to use and set it up
  myPropSet = new PropertiesSet();

  // Create the sound object; the sound subsystem isn't actually
  // opened until needed, so this is non-blocking (on those systems
  // that only have a single sound device (no hardware mixing)
  createSound();
  
  // Seed RNG. This will likely get re-called, e.g. by the ALEInterface, but is needed
  // by other interfaces.
  resetRNGSeed();

  return true;
}

void OSystem::resetRNGSeed() {

  // We seed the random number generator. The 'time' seed is somewhat redundant, since the
  // rng defaults to time. But we'll do it anyway.
  if (mySettings->getInt("random_seed") == 0) {
    myRandGen.seed((uint32_t)time(NULL));
  } else {
    int seed = mySettings->getInt("random_seed");
    assert(seed >= 0);
    myRandGen.seed((uint32_t)seed);
  }
}

bool OSystem::saveState(Serializer& out) {

    // Here we serialize the RNG state.
    return myRandGen.saveState(out);
}

bool OSystem::loadState(Deserializer& in) {
    return myRandGen.loadState(in);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void OSystem::setFramerate(uint32_t framerate)
{
  myDisplayFrameRate = framerate;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void OSystem::createSound()
{
  if (mySound != NULL) {
    delete mySound;
  }
  mySound = NULL;

#ifdef SOUND_SUPPORT
  // If requested (& supported), enable sound
  if (mySettings->getBool("sound") == true) {
      mySound = new SoundSDL(mySettings);
      mySound->initialize();
  }
  else {
      mySound = new SoundNull(mySettings);
  }
#else
  mySettings->setBool("sound", false);
  mySound = new SoundNull(mySettings);
#endif
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool OSystem::createConsole(const fs::path& romfile)
{
  // Do a little error checking; it shouldn't be necessary
  if(myConsole) deleteConsole();

  bool retval = false; 

  // If a blank ROM has been given, we reload the current one (assuming one exists)
  if (romfile.empty()) {
    if (myRomFile.empty()) {
      ale::Logger::Error << "ERROR: Rom file not specified ..." << std::endl;
      return false;
    }
  }
  else
    myRomFile = romfile.string();

  // Open the cartridge image and read it in
  uint8_t* image = nullptr;
  int size = -1;
  std::string md5;
  if(openROM(myRomFile, md5, &image, &size))
  {
    // Get all required info for creating a valid console
    Cartridge* cart = nullptr;
    Properties props;
    if(queryConsoleInfo(image, size, md5, &cart, props))
    {
      // Create an instance of the 2600 game console
      myConsole = new Console(this, cart, props);

      ale::Logger::Info << "Game console created:" << std::endl
            << "  ROM file:  " << myRomFile << std::endl
            << myConsole->about() << std::endl;

      retval = true;
    }
    else
    {
      ale::Logger::Error << "ERROR: Couldn't create console for " << myRomFile << " ..." << std::endl;
      retval = false;
    }
  }
  else
  {
    ale::Logger::Error << "ERROR: Couldn't open " << myRomFile << " ..." << std::endl;
    retval = false;
  }

  // Free the image since we don't need it any longer
  delete[] image;

  if (mySettings->getBool("display_screen", true)) {
#ifndef __USE_SDL
    ale::Logger::Error << "Screen display requires directive __USE_SDL to be defined."
                            << " Please recompile with flag '-D__USE_SDL'."
                            << " See makefile for more information."
                            << std::endl;
    exit(1);
#endif
    p_display_screen = new ale::DisplayScreen(&myConsole->mediaSource(),
                                              mySound, m_colour_palette); 
  }

  return retval;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void OSystem::deleteConsole()
{
  if(myConsole)
  {
    mySound->close();
    delete myConsole;  
    myConsole = NULL;
  }
  if (p_display_screen) {       //ALE
    delete p_display_screen;    //ALE
    p_display_screen = NULL;    //ALE 
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool OSystem::openROM(const fs::path& rom, std::string& md5, uint8_t** image, int* size)
{
  // Assume the file is either gzip'ed or not compressed at all
  gzFile f = gzopen(rom.string().c_str(), "rb");
  if(!f)
    return false;

  *image = new uint8_t[MAX_ROM_SIZE];
  *size = gzread(f, *image, MAX_ROM_SIZE);
  gzclose(f);

  // If we get to this point, we know we have a valid file to open
  // Now we make sure that the file has a valid properties entry
  md5 = MD5(*image, *size);

  // Some games may not have a name, since there may not
  // be an entry in stella.pro.  In that case, we use the rom name
  // and reinsert the properties object
  Properties props;
  myPropSet->getMD5(md5, props);

  std::string name = props.get(Cartridge_Name);
  if(name == "Untitled")
  {
    // Use the filename stem if we don't have this ROM in DefProps.
    // Stem is just the filename excluding the extension.
    // ROM is a valid file so we don't have to do extensive checks here
    fs::path rom_path(rom);
    props.set(Cartridge_MD5, md5);
    props.set(Cartridge_Name, rom_path.stem().string());
    myPropSet->insert(props, false);
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool OSystem::queryConsoleInfo(const uint8_t* image, uint32_t size,
                               const std::string& md5,
                               Cartridge** cart, Properties& props)
{
  // Get a valid set of properties, including any entered on the commandline
  std::string s;
  myPropSet->getMD5(md5, props);
  
    s = mySettings->getString("type");
    if(s != "") props.set(Cartridge_Type, s);
    s = mySettings->getString("channels");
    if(s != "") props.set(Cartridge_Sound, s);
    s = mySettings->getString("ld");
    if (s == "A") {
        ale::Logger::Info << "Setting Left Player's Difficulty to mode: A" << std::endl;
    }
    if(s != "") props.set(Console_LeftDifficulty, s);
    s = mySettings->getString("rd");
    if(s != "") props.set(Console_RightDifficulty, s);
    s = mySettings->getString("tv");
    if(s != "") props.set(Console_TelevisionType, s);
    s = mySettings->getString("sp");
    if(s != "") props.set(Console_SwapPorts, s);
    s = mySettings->getString("lc");
    if(s != "") props.set(Controller_Left, s);
    s = mySettings->getString("rc");
    if(s != "") props.set(Controller_Right, s);
    s = mySettings->getString("bc");
    if(s != "") { props.set(Controller_Left, s); props.set(Controller_Right, s); }
    s = mySettings->getString("cp");
    if(s != "") props.set(Controller_SwapPaddles, s);
    s = mySettings->getString("format");
    if(s != "") props.set(Display_Format, s);
    s = mySettings->getString("ystart");
    if(s != "") props.set(Display_YStart, s);
    s = mySettings->getString("height");
    if(s != "") props.set(Display_Height, s);
    s = mySettings->getString("pp");
    if(s != "") props.set(Display_Phosphor, s);
    s = mySettings->getString("ppblend");
    if(s != "") props.set(Display_PPBlend, s);
    s = mySettings->getString("hmove");
    if(s != "") props.set(Emulation_HmoveBlanks, s);

  *cart = Cartridge::create(image, size, props, *mySettings, myRandGen);
  if(!*cart)
    return false;

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
OSystem::OSystem(const OSystem& osystem)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
OSystem& OSystem::operator = (const OSystem&)
{
  assert(false);

  return *this;
}
