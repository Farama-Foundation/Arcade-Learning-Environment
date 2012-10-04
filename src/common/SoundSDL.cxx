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
// $Id: SoundSDL.cxx,v 1.38 2007/07/27 13:49:16 stephena Exp $
//============================================================================

#ifdef SOUND_SUPPORT

#include <sstream>
#include <cassert>
#include <cmath>
#include <SDL/SDL.h>

#include "TIASnd.hxx"
#include "FrameBuffer.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"
#include "Settings.hxx"
#include "System.hxx"
#include "OSystem.hxx"

#include "SoundSDL.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SoundSDL::SoundSDL(OSystem* osystem)
  : Sound(osystem),
    myIsEnabled(osystem->settings().getBool("sound")),
    myIsInitializedFlag(false),
    myLastRegisterSetCycle(0),
    myDisplayFrameRate(60),
    myNumChannels(1),
    myFragmentSizeLogBase2(0),
    myIsMuted(false),
    myVolume(100)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SoundSDL::~SoundSDL()
{
  // Close the SDL audio system if it's initialized
  close();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::setEnabled(bool state)
{
  myIsEnabled = state;
  myOSystem->settings().setBool("sound", state);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::initialize()
{
  // Check whether to start the sound subsystem
  if(!myIsEnabled)
  {
    close();
    if(myOSystem->settings().getBool("showinfo"))
      cerr << "Sound disabled." << endl << endl;
    return;
  }

  // Make sure the sound queue is clear
  myRegWriteQueue.clear();
  myTIASound.reset();

  if(!((SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO) > 0))
  {
    myIsInitializedFlag = false;
    myIsMuted = false;
    myLastRegisterSetCycle = 0;

    if(SDL_InitSubSystem(SDL_INIT_AUDIO) < 0)
    {
      cerr << "WARNING: Couldn't initialize SDL audio system! " << endl;
      cerr << "         " << SDL_GetError() << endl;
      return;
    }
    else
    {
      uInt32 fragsize = myOSystem->settings().getInt("fragsize");
      Int32 frequency = myOSystem->settings().getInt("freq");
      Int32 tiafreq   = myOSystem->settings().getInt("tiafreq");

      SDL_AudioSpec desired;
      desired.freq   = frequency;
    #ifndef GP2X
      desired.format = AUDIO_U8;
    #else
      desired.format = AUDIO_U16;
    #endif
      desired.channels = myNumChannels;
      desired.samples  = fragsize;
      desired.callback = callback;
      desired.userdata = (void*)this;

      if(SDL_OpenAudio(&desired, &myHardwareSpec) < 0)
      {
        cerr << "WARNING: Couldn't open SDL audio system! " << endl;
        cerr << "         " << SDL_GetError() << endl;
        return;
      }

      // Make sure the sample buffer isn't to big (if it is the sound code
      // will not work so we'll need to disable the audio support)
      if(((float)myHardwareSpec.samples / (float)myHardwareSpec.freq) >= 0.25)
      {
        cerr << "WARNING: Sound device doesn't support realtime audio! Make ";
        cerr << "sure a sound" << endl;
        cerr << "         server isn't running.  Audio is disabled." << endl;

        SDL_CloseAudio();
        return;
      }

      myIsInitializedFlag = true;
      myIsMuted = false;
      myFragmentSizeLogBase2 = log((double)myHardwareSpec.samples) / log(2.0);

        /*
        cerr << "Freq: " << (int)myHardwareSpec.freq << endl;
        cerr << "Format: " << (int)myHardwareSpec.format << endl;
        cerr << "Channels: " << (int)myHardwareSpec.channels << endl;
        cerr << "Silence: " << (int)myHardwareSpec.silence << endl;
        cerr << "Samples: " << (int)myHardwareSpec.samples << endl;
        cerr << "Size: " << (int)myHardwareSpec.size << endl;
          */

      // Now initialize the TIASound object which will actually generate sound
      myTIASound.outputFrequency(myHardwareSpec.freq);
      myTIASound.tiaFrequency(tiafreq);
      myTIASound.channels(myHardwareSpec.channels);

      bool clipvol = myOSystem->settings().getBool("clipvol");
      myTIASound.clipVolume(clipvol);

      // Adjust volume to that defined in settings
      myVolume = myOSystem->settings().getInt("volume");
      setVolume(myVolume);

      // Show some info
      if(myOSystem->settings().getBool("showinfo"))
        cerr << "Sound enabled:"  << endl
             << "  Volume     : " << myVolume << endl
             << "  Frag size  : " << fragsize << endl
             << "  Frequency  : " << myHardwareSpec.freq << endl
             << "  Format     : " << myHardwareSpec.format << endl
             << "  TIA Freq.  : " << tiafreq << endl
             << "  Channels   : " << myNumChannels << endl
             << "  Clip volume: " << (int)clipvol << endl << endl;
    }
  }

  // And start the SDL sound subsystem ...
  if(myIsInitializedFlag)
  {
    SDL_PauseAudio(0);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::close()
{
  if(myIsInitializedFlag)
  {
    SDL_CloseAudio();
    myIsInitializedFlag = false;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool SoundSDL::isSuccessfullyInitialized() const
{
  return myIsInitializedFlag;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::mute(bool state)
{
  if(myIsInitializedFlag)
  {
    // Ignore multiple calls to do the same thing
    if(myIsMuted == state)
    {
      return;
    }

    myIsMuted = state;

    SDL_PauseAudio(myIsMuted ? 1 : 0);
    myRegWriteQueue.clear();
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::reset()
{
  if(myIsInitializedFlag)
  {
    SDL_PauseAudio(1);
    myIsMuted = false;
    myLastRegisterSetCycle = 0;
    myTIASound.reset();
    myRegWriteQueue.clear();
    SDL_PauseAudio(0);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::setVolume(Int32 percent)
{
  if(myIsInitializedFlag)
  {
    if((percent >= 0) && (percent <= 100))
    {
      myOSystem->settings().setInt("volume", percent);
      SDL_LockAudio();
      myVolume = percent;
      myTIASound.volume(percent);
      SDL_UnlockAudio();
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::adjustVolume(Int8 direction)
{
  ostringstream strval;
  string message;

  Int32 percent = myVolume;

  if(direction == -1)
    percent -= 2;
  else if(direction == 1)
    percent += 2;

  if((percent < 0) || (percent > 100))
    return;

  setVolume(percent);

  // Now show an onscreen message
  strval << percent;
  message = "Volume set to ";
  message += strval.str();

  myOSystem->frameBuffer().showMessage(message);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::adjustCycleCounter(Int32 amount)
{
  myLastRegisterSetCycle += amount;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::setChannels(uInt32 channels)
{
  if(channels == 1 || channels == 2)
    myNumChannels = channels;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::setFrameRate(uInt32 framerate)
{
  // FIXME, we should clear out the queue or adjust the values in it
  myDisplayFrameRate = framerate;
  myLastRegisterSetCycle = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::set(uInt16 addr, uInt8 value, Int32 cycle)
{
  SDL_LockAudio();

  // First, calulate how many seconds would have past since the last
  // register write on a real 2600
  double delta = (((double)(cycle - myLastRegisterSetCycle)) / 
      (1193191.66666667));

  // Now, adjust the time based on the frame rate the user has selected. For
  // the sound to "scale" correctly, we have to know the games real frame 
  // rate (e.g., 50 or 60) and the currently emulated frame rate. We use these
  // values to "scale" the time before the register change occurs.
  delta = delta * (myDisplayFrameRate / (double)myOSystem->frameRate());
  RegWrite info;
  info.addr = addr;
  info.value = value;
  info.delta = delta;
  myRegWriteQueue.enqueue(info);

  // Update last cycle counter to the current cycle
  myLastRegisterSetCycle = cycle;

  SDL_UnlockAudio();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::processFragment(uInt8* stream, Int32 length)
{
  if(!myIsInitializedFlag)
    return;

  uInt32 channels = myHardwareSpec.channels;
  length = length / channels;

  // If there are excessive items on the queue then we'll remove some
  if(myRegWriteQueue.duration() > 
      (myFragmentSizeLogBase2 / myDisplayFrameRate))
  {
    double removed = 0.0;
    while(removed < ((myFragmentSizeLogBase2 - 1) / myDisplayFrameRate))
    {
      RegWrite& info = myRegWriteQueue.front();
      removed += info.delta;
      myTIASound.set(info.addr, info.value);
      myRegWriteQueue.dequeue();
    }
//    cerr << "Removed Items from RegWriteQueue!" << endl;
  }

  double position = 0.0;
  double remaining = length;

  while(remaining > 0.0)
  {
    if(myRegWriteQueue.size() == 0)
    {
      // There are no more pending TIA sound register updates so we'll
      // use the current settings to finish filling the sound fragment
//    myTIASound.process(stream + (uInt32)position, length - (uInt32)position);
      myTIASound.process(stream + ((uInt32)position * channels),
          length - (uInt32)position);

      // Since we had to fill the fragment we'll reset the cycle counter
      // to zero.  NOTE: This isn't 100% correct, however, it'll do for
      // now.  We should really remember the overrun and remove it from
      // the delta of the next write.
      myLastRegisterSetCycle = 0;
      break;
    }
    else
    {
      // There are pending TIA sound register updates so we need to
      // update the sound buffer to the point of the next register update
      RegWrite& info = myRegWriteQueue.front();

      // How long will the remaining samples in the fragment take to play
      double duration = remaining / (double)myHardwareSpec.freq;

      // Does the register update occur before the end of the fragment?
      if(info.delta <= duration)
      {
        // If the register update time hasn't already passed then
        // process samples upto the point where it should occur
        if(info.delta > 0.0)
        {
          // Process the fragment upto the next TIA register write.  We
          // round the count passed to process up if needed.
          double samples = (myHardwareSpec.freq * info.delta);
//        myTIASound.process(stream + (uInt32)position, (uInt32)samples +
//            (uInt32)(position + samples) - 
//            ((uInt32)position + (uInt32)samples));
          myTIASound.process(stream + ((uInt32)position * channels),
              (uInt32)samples + (uInt32)(position + samples) - 
              ((uInt32)position + (uInt32)samples));

          position += samples;
          remaining -= samples;
        }
        myTIASound.set(info.addr, info.value);
        myRegWriteQueue.dequeue();
      }
      else
      {
        // The next register update occurs in the next fragment so finish
        // this fragment with the current TIA settings and reduce the register
        // update delay by the corresponding amount of time
//      myTIASound.process(stream + (uInt32)position, length - (uInt32)position);
        myTIASound.process(stream + ((uInt32)position * channels),
            length - (uInt32)position);
        info.delta -= duration;
        break;
      }
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::callback(void* udata, uInt8* stream, int len)
{
  SoundSDL* sound = (SoundSDL*)udata;
  sound->processFragment(stream, (Int32)len);
#ifdef ATARIVOX_SUPPORT
  cerr << "SoundSDL::callback(): len==" << len << endl;

  // See if we need sound from the AtariVox
  AtariVox *vox = sound->myOSystem->console().atariVox();
  if(vox)
  {
    // If so, mix 'em together (this is a crappy way to mix audio streams...)
    uInt8 *s = stream;
    for(int i=0; i<len/OUTPUT_BUFFER_SIZE; i++)
    {
      int count;
      uInt8 *voxSamples = vox->getSpeakJet()->getSamples(&count);
      if(!count)
        break;
      SDL_MixAudio(s, voxSamples, OUTPUT_BUFFER_SIZE, SDL_MIX_MAXVOLUME);
      s += OUTPUT_BUFFER_SIZE;
    }
  }
#endif
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool SoundSDL::load(Deserializer& in)
{
  string device = "TIASound";

  try
  {
    if(in.getString() != device)
      return false;

    uInt8 reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0, reg6 = 0;
    reg1 = (uInt8) in.getInt();
    reg2 = (uInt8) in.getInt();
    reg3 = (uInt8) in.getInt();
    reg4 = (uInt8) in.getInt();
    reg5 = (uInt8) in.getInt();
    reg6 = (uInt8) in.getInt();

    myLastRegisterSetCycle = (Int32) in.getInt();

    // Only update the TIA sound registers if sound is enabled
    // Make sure to empty the queue of previous sound fragments
    if(myIsInitializedFlag)
    {
      SDL_PauseAudio(1);
      myRegWriteQueue.clear();
      myTIASound.set(0x15, reg1);
      myTIASound.set(0x16, reg2);
      myTIASound.set(0x17, reg3);
      myTIASound.set(0x18, reg4);
      myTIASound.set(0x19, reg5);
      myTIASound.set(0x1a, reg6);
      SDL_PauseAudio(0);
    }
  }
  catch(char *msg)
  {
    cerr << msg << endl;
    return false;
  }
  catch(...)
  {
    cerr << "Unknown error in load state for " << device << endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool SoundSDL::save(Serializer& out)
{
  string device = "TIASound";

  try
  {
    out.putString(device);

    uInt8 reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0, reg6 = 0;

    // Only get the TIA sound registers if sound is enabled
    if(myIsInitializedFlag)
    {
      reg1 = myTIASound.get(0x15);
      reg2 = myTIASound.get(0x16);
      reg3 = myTIASound.get(0x17);
      reg4 = myTIASound.get(0x18);
      reg5 = myTIASound.get(0x19);
      reg6 = myTIASound.get(0x1a);
    }

    out.putInt(reg1);
    out.putInt(reg2);
    out.putInt(reg3);
    out.putInt(reg4);
    out.putInt(reg5);
    out.putInt(reg6);

    out.putInt(myLastRegisterSetCycle);
  }
  catch(char *msg)
  {
    cerr << msg << endl;
    return false;
  }
  catch(...)
  {
    cerr << "Unknown error in save state for " << device << endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SoundSDL::RegWriteQueue::RegWriteQueue(uInt32 capacity)
  : myCapacity(capacity),
    myBuffer(0),
    mySize(0),
    myHead(0),
    myTail(0)
{
  myBuffer = new RegWrite[myCapacity];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SoundSDL::RegWriteQueue::~RegWriteQueue()
{
  delete[] myBuffer;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::RegWriteQueue::clear()
{
  myHead = myTail = mySize = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::RegWriteQueue::dequeue()
{
  if(mySize > 0)
  {
    myHead = (myHead + 1) % myCapacity;
    --mySize;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
double SoundSDL::RegWriteQueue::duration()
{
  double duration = 0.0;
  for(uInt32 i = 0; i < mySize; ++i)
  {
    duration += myBuffer[(myHead + i) % myCapacity].delta;
  }
  return duration;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::RegWriteQueue::enqueue(const RegWrite& info)
{
  // If an attempt is made to enqueue more than the queue can hold then
  // we'll enlarge the queue's capacity.
  if(mySize == myCapacity)
  {
    grow();
  }

  myBuffer[myTail] = info;
  myTail = (myTail + 1) % myCapacity;
  ++mySize;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SoundSDL::RegWrite& SoundSDL::RegWriteQueue::front()
{
  assert(mySize != 0);
  return myBuffer[myHead];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 SoundSDL::RegWriteQueue::size() const
{
  return mySize;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundSDL::RegWriteQueue::grow()
{
  RegWrite* buffer = new RegWrite[myCapacity * 2];
  for(uInt32 i = 0; i < mySize; ++i)
  {
    buffer[i] = myBuffer[(myHead + i) % myCapacity];
  }
  myHead = 0;
  myTail = mySize;
  myCapacity = myCapacity * 2;
  delete[] myBuffer;
  myBuffer = buffer;
}

#endif  // SOUND_SUPPORT
