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
// $Id: SoundNull.hxx,v 1.6 2007/01/01 18:04:40 stephena Exp $
//============================================================================

#ifndef SOUND_NULL_HXX
#define SOUND_NULL_HXX

class OSystem;
class Serializer;
class Deserializer;

#include "../emucore/m6502/src/bspf/src/bspf.hxx"
#include "../emucore/Sound.hxx"

/**
  This class implements a Null sound object, where-by sound generation
  is completely disabled.

  @author Stephen Anthony
  @version $Id: SoundNull.hxx,v 1.6 2007/01/01 18:04:40 stephena Exp $
*/
class SoundNull : public Sound
{
  public:
    /**
      Create a new sound object.  The init method must be invoked before
      using the object.
    */
    SoundNull(OSystem* osystem);
 
    /**
      Destructor
    */
    virtual ~SoundNull();

  public: 
    /**
      Enables/disables the sound subsystem.

      @param enable  Either true or false, to enable or disable the sound system
      @return        Whether the sound system was enabled or disabled
    */
    void setEnabled(bool enable) { }

    /**
      The system cycle counter is being adjusting by the specified amount.  Any
      members using the system cycle counter should be adjusted as needed.

      @param amount The amount the cycle counter is being adjusted by
    */
    void adjustCycleCounter(Int32 amount) { }

    /**
      Sets the number of channels (mono or stereo sound).

      @param channels The number of channels
    */
    void setChannels(uInt32 channels) { }

    /**
      Sets the display framerate.  Sound generation for NTSC and PAL games
      depends on the framerate, so we need to set it here.

      @param framerate The base framerate depending on NTSC or PAL ROM
    */
    void setFrameRate(uInt32 framerate) { }

    /**
      Initializes the sound device.  This must be called before any
      calls are made to derived methods.
    */
    void initialize() { }

    /**
      Should be called to close the sound device.  Once called the sound
      device can be started again using the initialize method.
    */
    void close() { }

    /**
      Return true iff the sound device was successfully initialized.

      @return true iff the sound device was successfully initialized.
    */
    bool isSuccessfullyInitialized() const { return false; }

    /**
      Set the mute state of the sound object.  While muted no sound is played.

      @param state Mutes sound if true, unmute if false
    */
    void mute(bool state) { }

    /**
      Reset the sound device.
    */
    void reset() { }

    /**
      Sets the sound register to a given value.

      @param addr  The register address
      @param value The value to save into the register
      @param cycle The system cycle at which the register is being updated
    */
    void set(uInt16 addr, uInt8 value, Int32 cycle) { }

    /**
      Sets the volume of the sound device to the specified level.  The
      volume is given as a percentage from 0 to 100.  Values outside
      this range indicate that the volume shouldn't be changed at all.

      @param percent The new volume percentage level for the sound device
    */
    void setVolume(Int32 percent) { }

    /**
      Adjusts the volume of the sound device based on the given direction.

      @param direction  Increase or decrease the current volume by a predefined
                        amount based on the direction (1 = increase, -1 =decrease)
    */
    void adjustVolume(Int8 direction) { }

public:
    /**
      Loads the current state of this device from the given Deserializer.

      @param in The deserializer device to load from.
      @return The result of the load.  True on success, false on failure.
    */
    bool load(Deserializer& in);

    /**
      Saves the current state of this device to the given Serializer.

      @param out The serializer device to save to.
      @return The result of the save.  True on success, false on failure.
    */
    bool save(Serializer& out);
};

#endif
