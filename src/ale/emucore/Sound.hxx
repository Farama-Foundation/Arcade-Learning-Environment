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
// $Id: Sound.hxx,v 1.23 2007/01/01 18:04:50 stephena Exp $
//============================================================================

#ifndef SOUND_HXX
#define SOUND_HXX

#include <cstdint>

namespace ale {
namespace stella {

class Settings;
class Serializer;
class Deserializer;


/**
  This class is an abstract base class for the various sound objects.
  It has no functionality whatsoever.

  @author Stephen Anthony
  @version $Id: Sound.hxx,v 1.23 2007/01/01 18:04:50 stephena Exp $
*/
class Sound
{
  public:
    /**
      Create a new sound object.  The init method must be invoked before
      using the object.
    */
    Sound(Settings* settings) { mySettings = settings; }

    /**
      Destructor
    */
    virtual ~Sound() { };

  public:
    /**
      Enables/disables the sound subsystem.

      @param enable  Either true or false, to enable or disable the sound system
    */
    virtual void setEnabled(bool enable) = 0;

    /**
      The system cycle counter is being adjusting by the specified amount.  Any
      members using the system cycle counter should be adjusted as needed.

      @param amount The amount the cycle counter is being adjusted by
    */
    virtual void adjustCycleCounter(int amount) = 0;

    /**
      Sets the number of channels (mono or stereo sound).

      @param channels The number of channels
    */
    virtual void setChannels(uint32_t channels) = 0;

    /**
      Sets the display framerate.  Sound generation for NTSC and PAL games
      depends on the framerate, so we need to set it here.

      @param framerate The base framerate depending on NTSC or PAL ROM
    */
    virtual void setFrameRate(uint32_t framerate) = 0;

    /**
      Initializes the sound device.  This must be called before any
      calls are made to derived methods.
    */
    virtual void initialize() = 0;

    /**
      Should be called to close the sound device.  Once called the sound
      device can be started again using the initialize method.
    */
    virtual void close() = 0;

    /**
      Return true iff the sound device was successfully initialized.

      @return true iff the sound device was successfully initialized.
    */
    virtual bool isSuccessfullyInitialized() const = 0;

    /**
      Set the mute state of the sound object.  While muted no sound is played.

      @param state Mutes sound if true, unmute if false
    */
    virtual void mute(bool state) = 0;

    /**
      Reset the sound device.
    */
    virtual void reset() = 0;

    /**
      Sets the sound register to a given value.

      @param addr  The register address
      @param value The value to save into the register
      @param cycle The system cycle at which the register is being updated
    */
    virtual void set(uint16_t addr, uint8_t value, int cycle) = 0;

    /**
      Sets the volume of the sound device to the specified level.  The
      volume is given as a percentage from 0 to 100.  Values outside
      this range indicate that the volume shouldn't be changed at all.

      @param percent The new volume percentage level for the sound device
    */
    virtual void setVolume(int percent) = 0;

    /**
      Adjusts the volume of the sound device based on the given direction.

      @param direction  Increase or decrease the current volume by a predefined
                        amount based on the direction (1 = increase, -1 =decrease)
    */
    virtual void adjustVolume(int8_t direction) = 0;

    /**
      * Tells the sound engine to record one frame's worth of sound.
      */
    virtual void recordNextFrame() = 0;

    /**
      * Processes audio for raw sample generation (applies all reg updates, fills buffer)
      */
    virtual void process(uint8_t* buffer, uint32_t samples) = 0;

public:
    /**
      Loads the current state of this device from the given Deserializer.

      @param in The deserializer device to load from.
      @return The result of the load.  True on success, false on failure.
    */
    virtual bool load(Deserializer& in) = 0;

    /**
      Saves the current state of this device to the given Serializer.

      @param out The serializer device to save to.
      @return The result of the save.  True on success, false on failure.
    */
    virtual bool save(Serializer& out) = 0;

  protected:
    // The emulator Settings
    Settings* mySettings;
};

}  // namespace stella
}  // namespace ale

#endif
