/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *   Matthew Hausknecht and the Reinforcement Learning and Artificial Intelligence
 *   Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  SoundRaw.hxx
 *
 *  A class for generating raw Atari 2600 sound samples.
 *
 **************************************************************************** */

#ifndef SOUND_RAW_HXX
#define SOUND_RAW_HXX

namespace ale {
namespace stella {

class Settings;
class Serializer;
class Deserializer;

}  // namespace stella
}  // namespace ale

#include "ale/emucore/Sound.hxx"
#include "ale/emucore/TIASnd.hxx"
#include <deque>

namespace ale {

/**
  This class implements a sound object for generating per-frame sound data
  from raw TIA sound samples.
*/
class SoundRaw : public stella::Sound
{
  public:
    // The Atari 2600 specification lists the audio clock as ~30KHz. With
    // a default display rate of 60 FPS, we can expect 512 samples per frame.
    static constexpr int SamplesPerFrame = 512;

    /**
      Create a new sound object.  The init method must be invoked before
      using the object.
    */
    SoundRaw(stella::Settings* settings);

    /**
      Destructor
    */
    virtual ~SoundRaw();

  public:
    /**
      Enables/disables the sound subsystem.

      @param enable  Either true or false, to enable or disable the sound system
      @return        Whether the sound system was enabled or disabled
    */
    void setEnabled(bool);

    /**
      The system cycle counter is being adjusting by the specified amount.  Any
      members using the system cycle counter should be adjusted as needed.

      @param amount The amount the cycle counter is being adjusted by
    */
    void adjustCycleCounter(int);

    /**
      Sets the number of channels (mono or stereo sound).
      Currently, only mono is supposed for raw sample generation.

      @param channels The number of channels
    */
    void setChannels(uint32_t) { }

    /**
      Sets the display framerate.  Sound generation for NTSC and PAL games
      depends on the framerate, so we need to set it here.

      @param framerate The base framerate depending on NTSC or PAL ROM
    */
    void setFrameRate(uint32_t) { }

    /**
      Initializes the sound device.  This must be called before any
      calls are made to derived methods.
    */
    void initialize();

    /**
      Should be called to close the sound device.  Once called the sound
      device can be started again using the initialize method.
    */
    void close();

    /**
      Return true iff the sound device was successfully initialized.

      @return true iff the sound device was successfully initialized.
    */
    bool isSuccessfullyInitialized() const;

    /**
      Set the mute state of the sound object.  While muted no sound is played.

      @param state Mutes sound if true, unmute if false
    */
    void mute(bool) { }

    /**
      Reset the sound device.
    */
    void reset();

    /**
      Sets the sound register to a given value.

      @param addr  The register address
      @param value The value to save into the register
      @param cycle The system cycle at which the register is being updated
    */
    void set(uint16_t, uint8_t, int);

    /**
      Sets the volume of the sound device to the specified level.  The
      volume is given as a percentage from 0 to 100.  Values outside
      this range indicate that the volume shouldn't be changed at all.

      @param percent The new volume percentage level for the sound device
    */
    void setVolume(int) { }

    /**
      Adjusts the volume of the sound device based on the given direction.

      @param direction  Increase or decrease the current volume by a predefined
                        amount based on the direction (1 = increase, -1 =decrease)
    */
    void adjustVolume(int8_t) { }

    /**
      * Tells the sound engine to record one frame's worth of sound.
      */
    virtual void recordNextFrame() { }

    /**
      * Processes audio for raw sample generation (applies all reg updates, fills buffer)
      */
    virtual void process(uint8_t* buffer, uint32_t samples);

public:
    /**
      Loads the current state of this device from the given Deserializer.

      @param in The deserializer device to load from.
      @return The result of the load.  True on success, false on failure.
    */
    bool load(stella::Deserializer& in);

    /**
      Saves the current state of this device to the given Serializer.

      @param out The serializer device to save to.
      @return The result of the save.  True on success, false on failure.
    */
    bool save(stella::Serializer& out);

protected:
    // Struct to hold information regarding a TIA sound register
    struct TIARegister
    {
      uint16_t addr;
      uint8_t value;
    };

private:
    // TIASound emulation object
    stella::TIASound myTIASound;

    // Indicates if the sound subsystem is to be initialized
    bool myIsEnabled;

    // Indicates if the sound device was successfully initialized
    bool myIsInitializedFlag;

    // Indicates the cycle when a sound register was last set
    int myLastRegisterSetCycle;

    // Queue of TIA register writes
    std::deque<TIARegister> myRegWriteQueue;
};

}  // namespace ale

#endif
