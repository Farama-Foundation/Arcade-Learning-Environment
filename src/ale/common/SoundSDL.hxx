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
// $Id: SoundSDL.hxx,v 1.18 2007/01/01 18:04:40 stephena Exp $
//============================================================================

#ifndef SOUND_SDL_HXX
#define SOUND_SDL_HXX

#ifdef SDL_SUPPORT

namespace ale {
namespace stella {

class Settings;

}  // namespace stella
}  // namespace ale

#include "ale/emucore/Sound.hxx"
#include "ale/emucore/MediaSrc.hxx"
#include "ale/emucore/TIASnd.hxx"

#include "ale/common/SDL2.hpp"
// If desired, we save sound to disk
#include "ale/common/SoundExporter.hpp"
#include <memory>

namespace ale {

/**
  This class implements the sound API for SDL.

  @author Stephen Anthony and Bradford W. Mott
  @version $Id: SoundSDL.hxx,v 1.18 2007/01/01 18:04:40 stephena Exp $
*/
class SoundSDL : public stella::Sound
{
  public:
    /**
      Create a new sound object.  The init method must be invoked before
      using the object.
    */
    SoundSDL(stella::Settings* settings);

    /**
      Destructor
    */
    virtual ~SoundSDL();

  public:
    /**
      Enables/disables the sound subsystem.

      @param state True or false, to enable or disable the sound system
    */
    void setEnabled(bool state);

    /**
      The system cycle counter is being adjusting by the specified amount. Any
      members using the system cycle counter should be adjusted as needed.

      @param amount The amount the cycle counter is being adjusted by
    */
    void adjustCycleCounter(int amount);

    /**
      Sets the number of channels (mono or stereo sound).

      @param channels The number of channels
    */
    void setChannels(uint32_t channels);

    /**
      Sets the display framerate.  Sound generation for NTSC and PAL games
      depends on the framerate, so we need to set it here.

      @param framerate The base framerate depending on NTSC or PAL ROM
    */
    void setFrameRate(uint32_t framerate);

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
    void mute(bool state);

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
    void set(uint16_t addr, uint8_t value, int cycle);

    /**
      Sets the volume of the sound device to the specified level.  The
      volume is given as a percentage from 0 to 100.  Values outside
      this range indicate that the volume shouldn't be changed at all.

      @param percent The new volume percentage level for the sound device
    */
    void setVolume(int percent);

    /**
      Adjusts the volume of the sound device based on the given direction.

      @param direction Increase or decrease the current volume by a predefined
          amount based on the direction (1 = increase, -1 = decrease)
    */
    void adjustVolume(int8_t direction);

    /**
      * Tells the sound engine to record one frame's worth of sound.
      */
    void recordNextFrame();

    /**
      * Processes audio for raw sample generation (applies all reg updates, fills buffer)
      */
    void process(uint8_t* buffer, uint32_t samples) { }

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
    /**
      Invoked by the sound callback to process the next sound fragment.

      @param stream Pointer to the start of the fragment
      @param length Length of the fragment
    */
    void processFragment(uint8_t* stream, int length);

  protected:
    // Struct to hold information regarding a TIA sound register write
    struct RegWrite
    {
      uint16_t addr;
      uint8_t value;
      double delta;
    };

    /**
      A queue class used to hold TIA sound register writes before being
      processed while creating a sound fragment.
    */
    class RegWriteQueue
    {
      public:
        /**
          Create a new queue instance with the specified initial
          capacity.  If the queue ever reaches its capacity then it will
          automatically increase its size.
        */
        RegWriteQueue(uint32_t capacity = 512);

        /**
          Destroy this queue instance.
        */
        virtual ~RegWriteQueue();

      public:
        /**
          Clear any items stored in the queue.
        */
        void clear();

        /**
          Dequeue the first object in the queue.
        */
        void dequeue();

        /**
          Return the duration of all the items in the queue.
        */
        double duration();

        /**
          Enqueue the specified object.
        */
        void enqueue(const RegWrite& info);

        /**
          Return the item at the front on the queue.

          @return The item at the front of the queue.
        */
        RegWrite& front();

        /**
          Answers the number of items currently in the queue.

          @return The number of items in the queue.
        */
        uint32_t size() const;

      private:
        // Increase the size of the queue
        void grow();

      private:
        uint32_t myCapacity;
        RegWrite* myBuffer;
        uint32_t mySize;
        uint32_t myHead;
        uint32_t myTail;
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

    // Indicates the base framerate depending on if the ROM is NTSC or PAL
    uint32_t myDisplayFrameRate;

    // Indicates the number of channels (mono or stereo)
    uint32_t myNumChannels;

    // Log base 2 of the selected fragment size
    double myFragmentSizeLogBase2;

    // Indicates if the sound is currently muted
    bool myIsMuted;

    // Current volume as a percentage (0 - 100)
    uint32_t myVolume;

    // Audio specification structure
    SDL_AudioSpec myHardwareSpec;

    // Queue of TIA register writes
    RegWriteQueue myRegWriteQueue;

  private:
    // Callback function invoked by the SDL Audio library when it needs data
    static void callback(void* udata, uint8_t* stream, int len);

    // Keeps track of how many samples we still need to record
    int myNumRecordSamplesNeeded;

    std::unique_ptr<ale::sound::SoundExporter> mySoundExporter;
};

}  // namespace ale

#endif  // SDL_SUPPORT
#endif
