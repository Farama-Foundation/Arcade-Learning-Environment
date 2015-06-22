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

#ifdef SOUND_SUPPORT

class OSystem;

#include "SDL.h"

#include "../emucore/Sound.hxx"
#include "../emucore/m6502/src/bspf/src/bspf.hxx"
#include "MediaSrc.hxx"
#include "TIASnd.hxx"

// If desired, we save sound to disk
#include "SoundExporter.hpp"
#include <memory>

/**
  This class implements the sound API for SDL.

  @author Stephen Anthony and Bradford W. Mott
  @version $Id: SoundSDL.hxx,v 1.18 2007/01/01 18:04:40 stephena Exp $
*/
class SoundSDL : public Sound
{
  public:
    /**
      Create a new sound object.  The init method must be invoked before
      using the object.
    */
    SoundSDL(OSystem* osystem);
 
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
    void adjustCycleCounter(Int32 amount);

    /**
      Sets the number of channels (mono or stereo sound).

      @param channels The number of channels
    */
    void setChannels(uInt32 channels);

    /**
      Sets the display framerate.  Sound generation for NTSC and PAL games
      depends on the framerate, so we need to set it here.

      @param framerate The base framerate depending on NTSC or PAL ROM
    */
    void setFrameRate(uInt32 framerate);

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
    void set(uInt16 addr, uInt8 value, Int32 cycle);

    /**
      Sets the volume of the sound device to the specified level.  The
      volume is given as a percentage from 0 to 100.  Values outside
      this range indicate that the volume shouldn't be changed at all.

      @param percent The new volume percentage level for the sound device
    */
    void setVolume(Int32 percent);

    /**
      Adjusts the volume of the sound device based on the given direction.

      @param direction Increase or decrease the current volume by a predefined
          amount based on the direction (1 = increase, -1 = decrease)
    */
    void adjustVolume(Int8 direction);

    /**
      * Tells the sound engine to record one frame's worth of sound.
      */
    void recordNextFrame(); 

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

  protected:
    /**
      Invoked by the sound callback to process the next sound fragment.

      @param stream Pointer to the start of the fragment
      @param length Length of the fragment
    */
    void processFragment(uInt8* stream, Int32 length);

  protected:
    // Struct to hold information regarding a TIA sound register write
    struct RegWrite
    {
      uInt16 addr;
      uInt8 value;
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
        RegWriteQueue(uInt32 capacity = 512);

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
        uInt32 size() const;

      private:
        // Increase the size of the queue
        void grow();

      private:
        uInt32 myCapacity;
        RegWrite* myBuffer;
        uInt32 mySize;
        uInt32 myHead;
        uInt32 myTail;
    };

  private:
    // TIASound emulation object
    TIASound myTIASound;

    // Indicates if the sound subsystem is to be initialized
    bool myIsEnabled;

    // Indicates if the sound device was successfully initialized
    bool myIsInitializedFlag;

    // Indicates the cycle when a sound register was last set
    Int32 myLastRegisterSetCycle;

    // Indicates the base framerate depending on if the ROM is NTSC or PAL
    uInt32 myDisplayFrameRate;

    // Indicates the number of channels (mono or stereo)
    uInt32 myNumChannels;

    // Log base 2 of the selected fragment size
    double myFragmentSizeLogBase2;

    // Indicates if the sound is currently muted
    bool myIsMuted;

    // Current volume as a percentage (0 - 100)
    uInt32 myVolume;

    // Audio specification structure
    SDL_AudioSpec myHardwareSpec;

    // Queue of TIA register writes
    RegWriteQueue myRegWriteQueue;

  private:
    // Callback function invoked by the SDL Audio library when it needs data
    static void callback(void* udata, uInt8* stream, int len);

    // Keeps track of how many samples we still need to record
    int myNumRecordSamplesNeeded; 

    std::auto_ptr<ale::sound::SoundExporter> mySoundExporter; 
};

#endif  // SOUND_SUPPORT
#endif
