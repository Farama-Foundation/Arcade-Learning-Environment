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
// $Id: SpeakJet.hxx,v 1.7 2007/01/01 18:04:50 stephena Exp $
//============================================================================

#ifdef ATARIVOX_SUPPORT

#ifndef SPEAKJET_HXX
#define SPEAKJET_HXX

/**
  Emulation of the Magnevation SpeakJet.
  This is the speech synthesizer chip used in the AtariVox.
  See AtariVox.hxx and .cxx for AtariVox specifics.

  This class doesn't attempt 100% accurate emulation of the SpeakJet,
  as the chip contains a proprietary algorithm that does some complex
  modelling (in other words, it doesn't just string samples together).
  For this emulation, I use a library called rsynth, which does something
  similar (models the human vocal/nasal tract), but is implemented
  in a totally different way. You might say I'm emulating the spirit
  of the SpeakJet, not the letter :)

  Implementation details:

  Both rsynth and the SpeakJet take a stream of phoneme codes and produce
  audio output.

  My SpeakJet class accepts the SpeakJet phonemes, one at a time, and
  translates them to rsynth phonemes (which are not quite one-to-one
  equivalent). As each phoneme is translated, it's added to a phoneme
  buffer.

  Because of the way rsynth is implemented, it needs a full word's worth
  of phonemes in its buffer before its speech function is called. This
  means I'll only call rsynth_phones() when I receive a SpeakJet code that
  indicates a pause, or end-of-word, or a control code (set parameters
  or such). This will result in a slight delay (typically, games will
  send one SJ code per frame).

  Also due to rsynth's implementation, I have to run it in a thread. This
  is because rsynth_phones() is a monolithic function that needs a string
  of phonemes, and takes a while to run (for the word "testing", it takes
  1/4 second on an Athlon 64 @ 1800MHz). We can't have the emulator pause
  for a quarter second while this happens, so I'll call rsynth_phones()
  in a separate thread, and have it fill a buffer from which our main
  thread will pull as much data as it needs. A typical word will be
  30-40 thousand samples, and we only need fragsize/2 samples at a time.

  As always when using threads, there will be locking in play...

  rsynth's output is always 16-bit samples. This class will have to
  convert them to 8-bit samples before feeding them to the SDL audio
  buffer.

  When using the AtariVox, we'll use SDL stereo sound. The regular TIA
  sound will come out the left channel, and the speech will come out
  the right. This isn't ideal, but it's the easiest way to mix the two
  (I don't want to add an SDL_mixer dependency). The real SpeakJet uses a
  separate speaker from the 2600 (the 2600 TIA sound comes from the TV,
  the SJ sound comes from a set of PC speakers), so splitting them to
  the left and right channels isn't unreasonable... However, it means
  no game can simultaneously use stereo sound and the AtariVox (for now,
  anyway).

  @author  B. Watson
  @version $Id: SpeakJet.hxx,v 1.7 2007/01/01 18:04:50 stephena Exp $
*/

#include "m6502/src/bspf/src/bspf.hxx"

#include <SDL.h>
#include <SDL_thread.h>
#include "rsynth/rsynth.h"

struct SpeechBuffer;


enum { INPUT_BUFFER_SIZE = 128 };
enum { OUTPUT_BUFFER_SIZE = 128 };
enum { SPEECH_BUFFERS = 1024 };
static SDL_sem *ourInputSemaphore;
static rsynth_t *rsynth;
static darray_t rsynthSamples;
// phonemeBuffer holds *translated* phonemes (e.g. rsynth phonemes,
// not SpeakJet phonemes).
static char phonemeBuffer[INPUT_BUFFER_SIZE];
// How many bytes are in the input buffer?
static uInt16 ourInputCount;


class SpeakJet
{
  public:
    /**
      Create a new SpeakJet with given buffer size. We use a circular linked
      list of fixed size, each node being a buffer of bufferSize 8-bit
      samples.

      @param bufferSize The size of each output buffer, presumably equal
                        to fragsize/2.
    */
    SpeakJet();

    ~SpeakJet();

  public:
    /**
      Writes a SpeakJet phoneme (or other code).
      These are the codes from page 16 of the Speaket User Manual.
      Not all codes are emulated. In particular, the non-speech noises
      (200 thru 254) will be treated as silence. Also, not all the
      control codes will actually work (will document later).

      @param code The SpeakJet code being written to the emulated chip
    */
    void write(uInt8 code);

    /**
      Returns a buffer full of 8-bit samples. This should be called every
      frame or so, or else the older buffers will get overwritten by new
      data.

      @param count This will be set to the number of samples that are
                   returned. Value ranges from 0 to bufferSize.
    */
    uInt8 *getSamples(int *count);

    /**
      Returns false if the phonemeBuffer is full, true otherwise.
    */
    bool chipReady();

  private:
    // function that spawns the rsynth thread
    void spawnThread();

    // function that the rsynth thread runs...
    // ...and it has to be a *function*, not a method, because SDL's
    // written in C. Dammit.
    static int thread(void *data);

  private:
    // These functions are called from the rsynth thread context only

    // speak() is our locking wrapper for rsynth_phones()
    static void speak();

    static void *save_sample(void *user_data,
                             float sample,
                             unsigned nsamp,
                             rsynth_t *rsynth);

    static void *flush_samples(void *user_data,
                               unsigned nsamp,
                               rsynth_t *rsynth);

    static short clip(long *clip_max, float input, float *peak);

  private:

    // True if last code was 20 thru 29
    bool needParameter;

    static const char *ourPhonemeTable[];

    SDL_Thread *ourThread;

    SpeechBuffer *myCurrentOutputBuffer;

    // We use this semaphore like so:
    // Main thread locks it initially
    // Main thread gathers up phonemes, storing in the input buffer,
    //  until it hits a pause/space,
    // then unlocks the semaphore.
    // The rsynth thread blocks on the semaphore until the main thread
    // is done feeding data into the buffer.
    // When the rsynth thread unblocks, it quickly copies the buffer to
    // a private buffer, then unlocks the semaphore so the main thread
    // can re-use the buffer.

    // Note to self: locking decrements the semaphore; unlocking increments
    // To lock (blocking): SDL_SemWait()
    // To unlock: SDL_SemPost()

    // Each output buffer also needs its own locking semaphore:
    // rsynth thread locks each buffer as it fills it, then unlocks it
    // when it's done, and moves on to the next buffer in the circular
    // list (blocking if it's locked).

    // When the main thread is ready to play audio, it grabs its idea
    // of what the next buffer is (blocking if it's locked), locks it, mixes
    // its contents with the TIA audio data (if it's not an empty buffer),
    // clears the buffer, then unlocks it.
    // Note that, if the rsynth thread has been sleeping a while, all
    // the buffers might be empty.

    // When the rsynth thread runs out of input, it should probably
    // listen on a condition, so it can be woken up when there's something
    // to do.

  private:
    // Convert a SpeakJet phoneme into one or more rsynth phonemes.
    // Input range is 0 to 255, but not all codes are supported yet.
    static const char *xlatePhoneme(uInt8 code);

};

// Where our output samples go.
struct SpeechBuffer
{
    SDL_sem *lock;
    SpeechBuffer *next;
    int items;
    uInt8 contents[OUTPUT_BUFFER_SIZE];
};

// For now, just a static array of them
static SpeechBuffer outputBuffers[SPEECH_BUFFERS];

static SpeechBuffer *ourCurrentWriteBuffer;
static uInt8 ourCurrentWritePosition;

#endif

#endif
