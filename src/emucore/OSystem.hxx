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

class PropertiesSet;
class GameController;
class Menu;
class CommandMenu;
class Launcher;
class Debugger;
class CheatManager;
class VideoDialog;

#include "common/Array.hxx"
#include "emucore/Sound.hxx"
#include "common/SoundNull.hxx"
#include "emucore/Settings.hxx"
#include "emucore/Console.hxx"
#include "emucore/Event.hxx"  //ALE 
#include "emucore/bspf/bspf.hxx"
#include "common/display_screen.h"
#include "common/ColourPalette.hpp"
#include "common/Log.hpp"

struct Resolution {
  uInt32 width;
  uInt32 height;
  std::string name;
};
typedef Common::Array<Resolution> ResolutionList;

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
    virtual ~OSystem();

    /**
      Create all child objects which belong to this OSystem
    */
    virtual bool create();

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
      Get the settings object of the system

      @return The settings object
    */
    inline Settings& settings() const { return *mySettings; }

    /**
      Get the set of game properties for the system

      @return The properties set object
    */
    inline PropertiesSet& propSet() const { return *myPropSet; }

    /**
      Get the console of the system.

      @return The console object
    */
    inline Console& console(void) const { return *myConsole; }

#ifdef DEBUGGER_SUPPORT
    /**
      Get the ROM debugger of the system.

      @return The debugger object
    */
    inline Debugger& debugger(void) const { return *myDebugger; }
#endif

#ifdef CHEATCODE_SUPPORT
    /**
      Get the cheat manager of the system.

      @return The cheatmanager object
    */
    inline CheatManager& cheat(void) const { return *myCheatManager; }
#endif

    /**
      Set the framerate for the video system.  It's placed in this class since
      the mainLoop() method is defined here.

      @param framerate  The video framerate to use
    */
    virtual void setFramerate(uInt32 framerate);

    /**
      Set all config file paths for the OSystem.
    */
    void setConfigPaths();

    /**
      Get the current framerate for the video system.

      @return  The video framerate currently in use
    */
    inline uInt32 frameRate() const { return myDisplayFrameRate; }

    /**
      Get the maximum dimensions of a window for the video hardware.
    */
    uInt32 desktopWidth() const  { return myDesktopWidth; }
    uInt32 desktopHeight() const { return myDesktopHeight; }

    /**
      Get the supported fullscreen resolutions for the video hardware.

      @return  An array of supported resolutions
    */
    const ResolutionList& supportedResolutions() const { return myResolutions; }

    /**
      Return the default directory for storing data.
    */
    const std::string& baseDir() const { return myBaseDir; }

    /**
      This method should be called to get the full path of the gamelist
      cache file (used by the Launcher to show a listing of available games).

      @return String representing the full path of the gamelist cache file.
    */
    const std::string& cacheFile() const { return myGameListCacheFile; }

    /**
      This method should be called to get the full path of the cheat file.

      @return String representing the full path of the cheat filename.
    */
    const std::string& cheatFile() const { return myCheatFile; }

    /**
      This method should be called to get the full path of the config file.

      @return String representing the full path of the config filename.
    */
    const std::string& configFile() const { return myConfigFile; }

    /**
      This method should be called to get the full path of the
      (optional) palette file.

      @return String representing the full path of the properties filename.
    */
    const std::string& paletteFile() const { return myPaletteFile; }

    /**
      This method should be called to get the full path of the
      properties file (stella.pro).

      @return String representing the full path of the properties filename.
    */
    const std::string& propertiesFile() const { return myPropertiesFile; }

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
    bool createConsole(const std::string& romfile = "");

    /**
      Deletes the currently defined console, if it exists.
      Also prints some statistics (fps, total frames, etc).
    */
    void deleteConsole();

    /**
      Gets all possible info about the ROM by creating a temporary
      Console object and querying it.

      @param romfile  The full pathname of the ROM to use
      @return  Some information about this ROM
    */
    std::string getROMInfo(const std::string& romfile);

    /**
      The features which are conditionally compiled into Stella.

      @return  The supported features
    */
    const std::string& features() const { return myFeatures; }

    /**
      Open the given ROM and return an array containing its contents.

      @param rom    The absolute pathname of the ROM file
      @param md5    The md5 calculated from the ROM file
      @param image  A pointer to store the ROM data
                    Note, the calling method is responsible for deleting this
      @param size   The amount of data read into the image array
      @return  False on any errors, else true
    */
    bool openROM(const std::string& rom, std::string& md5, uInt8** image, int* size);

    /**
      Issue a quit event to the OSystem.
    */
    void quit() { myQuitLoop = true; }

    void skipEmulation() { mySkipEmulation = true; }

    /**
      Returns the random number generator for this emulator.
    */
    Random& rng() { return myRandGen; }

    /**
      Resets the seed for our random number generator.
    */
    void resetRNGSeed();

    /** 
      Serializes the OSystem state.
    */
    bool saveState(Serializer& out);

    /** 
      Deserializes the OSystem state.
    */
    bool loadState(Deserializer& in);

  public:
    //////////////////////////////////////////////////////////////////////
    // The following methods are system-specific and must be implemented
    // in derived classes.
    //////////////////////////////////////////////////////////////////////
    /**
      This method returns number of ticks in microseconds.

      @return Current time in microseconds.
    */
    virtual uInt32 getTicks() = 0;

  protected:
    /**
      Set the base directory for all Stella files (these files may be
      located in other places through settings).
    */
    void setBaseDir(const std::string& basedir);

    /**
      Set the location of the gamelist cache file
    */
    void setCacheFile(const std::string& cachefile) { myGameListCacheFile = cachefile; }

    /**
      Set the locations of config file
    */
    void setConfigFile(const std::string& file) { myConfigFile = file; }



  protected:
    // Global Event object  //ALE 
    Event* myEvent;

    // Pointer to the Sound object
    Sound* mySound;

    // Pointer to the Settings object
    Settings* mySettings;

    // Pointer to the PropertiesSet object
    PropertiesSet* myPropSet;

    // Pointer to the (currently defined) Console object
    Console* myConsole;

    // Random number generator shared across the emulator's components
    Random myRandGen; 

    // Maximum dimensions of the desktop area
    uInt32 myDesktopWidth, myDesktopHeight;

    // Supported fullscreen resolutions
    ResolutionList myResolutions;

    // Number of times per second to iterate through the main loop
    uInt32 myDisplayFrameRate;

    // Indicates whether to stop the main loop
    bool myQuitLoop;

    // Indicates that the emulation should not occur on the next time step
    // This is reset to false after one step
    bool mySkipEmulation;

  private:
    enum { kNumUIPalettes = 2 };
    std::string myBaseDir;

    std::string myCheatFile;
    std::string myConfigFile;
    std::string myPaletteFile;
    std::string myPropertiesFile;

    std::string myGameListCacheFile;
    std::string myRomFile;

    std::string myFeatures;

    public: //ALE 
    // Time per frame for a video update, based on the current framerate
    uInt32 myTimePerFrame;

    // Indicates whether the main processing loop should proceed
    struct TimingInfo {
      uInt32 start;
      uInt32 current;
      uInt32 virt;
      uInt32 totalTime;
      uInt32 totalFrames;
    };
    TimingInfo myTimingInfo;

    ale::ColourPalette &colourPalette() { return m_colour_palette; }

  public:
    ale::DisplayScreen* p_display_screen; //MHAUSKN

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
    bool queryConsoleInfo(const uInt8* image, uInt32 size, const std::string& md5,
                          Cartridge** cart, Properties& props);

    /**
      Initializes the timing so that the mainloop is reset to its
      initial values.
    */
    void resetLoopTiming();

    // Copy constructor isn't supported by this class so make it private
    OSystem(const OSystem&);

    // Assignment operator isn't supported by this class so make it private
    OSystem& operator = (const OSystem&);
};

#endif
