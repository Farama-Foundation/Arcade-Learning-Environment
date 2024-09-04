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
// $Id: OSystem.hxx,v 1.55 2007/08/12 23:05:12 stephena Exp $
//============================================================================

#ifndef OSYSTEM_HXX
#define OSYSTEM_HXX

namespace ale {
namespace stella {

class PropertiesSet;

}  // namespace stella
}  // namespace ale

#include <filesystem>

#include "ale/emucore/Sound.hxx"
#include "ale/emucore/Screen.hxx"
#include "ale/common/SoundNull.hxx"
#include "ale/emucore/Settings.hxx"
#include "ale/emucore/Console.hxx"
#include "ale/emucore/Event.hxx"  //ALE
#include "ale/common/ColourPalette.hpp"
#include "ale/common/Log.hpp"

namespace fs = std::filesystem;

namespace ale {
namespace stella {

/**
  This class provides an interface for accessing operating system specific
  functions.  It also comprises an overall parent object, to which all the
  other objects belong.

  @author  Stephen Anthony
  @version $Id: OSystem.hxx,v 1.55 2007/08/12 23:05:12 stephena Exp $
*/
class OSystem
{
  public:
    /**
      Create a new OSystem abstract class
    */
    OSystem();

    /**
      Destructor
    */
    ~OSystem();

    /**
      Create all child objects which belong to this OSystem
    */
    bool create();

  public:
    /**
      Adds the specified settings object to the system.

      @param settings The settings object to add
    */
    void attach(Settings* settings) { mySettings = settings; }

    /**  //ALE
      Get the event object of the system

      @return The event object
    */
    inline Event* event() const { return myEvent; }

    /**
      Get the sound object of the system

      @return The sound object
    */
    inline Sound& sound() const { return *mySound; }

    /**
      Get the screen object of the system

      @return The screen object
    */
    inline Screen& screen() const { return *myScreen; }

    /**
      Get the settings object of the system

      @return The settings object
    */
    inline Settings& settings() const { return *mySettings; }

    /**
      Get the console of the system.

      @return The console object
    */
    inline Console& console(void) const { return *myConsole; }

    /**
      Set the framerate for the video system.  It's placed in this class since
      the mainLoop() method is defined here.

      @param framerate  The video framerate to use
    */
    void setFramerate(uint32_t framerate);

    /**
      Get the current framerate for the video system.

      @return  The video framerate currently in use
    */
    inline uint32_t frameRate() const { return myDisplayFrameRate; }

    /**
      This method should be called to get the full path of the currently
      loaded ROM.

      @return String representing the full path of the ROM file.
    */
    const std::string& romFile() const { return myRomFile; }

    /**
      Creates a new game console from the specified romfile.

      @param romfile  The full pathname of the ROM to use
      @return  True on successful creation, otherwise false
    */
    bool createConsole(const fs::path& romfile = "");

    /**
      Deletes the currently defined console, if it exists.
      Also prints some statistics (fps, total frames, etc).
    */
    void deleteConsole();

    /**
      Open the given ROM and return an array containing its contents.

      @param rom    The absolute pathname of the ROM file
      @param md5    The md5 calculated from the ROM file
      @param image  A pointer to store the ROM data
                    Note, the calling method is responsible for deleting this
      @param size   The amount of data read into the image array
      @return  False on any errors, else true
    */
    bool openROM(const fs::path& rom, std::string& md5, uint8_t** image, int* size);

  protected:
    // Global Event object  //ALE
    Event* myEvent;

    // Pointer to the Sound object
    Sound* mySound;

    // Pointer to the Screen object
    Screen* myScreen;

    // Pointer to the Settings object
    Settings* mySettings;

    // Pointer to the PropertiesSet object
    PropertiesSet* myPropSet;

    // Pointer to the (currently defined) Console object
    Console* myConsole;

    // Number of times per second to iterate through the main loop
    uint32_t myDisplayFrameRate;

  private:
    std::string myRomFile;

  public: //ALE
    ale::ColourPalette &colourPalette() { return m_colour_palette; }

  private:

    ale::ColourPalette m_colour_palette;

    /**
      Creates the various sound devices available in this system
      (for now, that means either 'SDL' or 'Null').
    */
    void createSound();

    /**
      Query valid info for creating a valid console.

      @return Success or failure for a valid console
    */
    bool queryConsoleInfo(const uint8_t* image, uint32_t size, const std::string& md5,
                          Cartridge** cart, Properties& props);

    // Copy constructor isn't supported by this class so make it private
    OSystem(const OSystem&);

    // Assignment operator isn't supported by this class so make it private
    OSystem& operator = (const OSystem&);
};

}  // namespace stella
}  // namespace ale

#endif
