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
// $Id: MediaSrc.hxx,v 1.17 2007/01/01 18:04:49 stephena Exp $
//============================================================================

#ifndef MEDIASOURCE_HXX
#define MEDIASOURCE_HXX


class MediaSource;
class Sound;

#include "m6502/src/bspf/src/bspf.hxx"

/**
  This class provides an interface for accessing graphics and audio data.

  @author  Bradford W. Mott
  @version $Id: MediaSrc.hxx,v 1.17 2007/01/01 18:04:49 stephena Exp $
*/
class MediaSource
{
  public:
    /**
      Create a new media source
    */
    MediaSource();
 
    /**
      Destructor
    */
    virtual ~MediaSource();

  public:
    /**
      This method should be called at an interval corresponding to the 
      desired frame rate to update the media source.  Invoking this method
      will update the graphics buffer and generate the corresponding audio
      samples.
    */
    virtual void update() = 0;

    /**
      Answers the current frame buffer

      @return Pointer to the current frame buffer
    */
    virtual uInt8* currentFrameBuffer() const = 0;

    /**
      Answers the previous frame buffer

      @return Pointer to the previous frame buffer
    */
    virtual uInt8* previousFrameBuffer() const = 0;

#ifdef DEBUGGER_SUPPORT
    /**
      This method should be called whenever a new scanline is to be drawn.
      Invoking this method will update the graphics buffer and generate
      the corresponding audio samples.
    */
    virtual void updateScanline() = 0;

    /**
      This method should be called whenever a new partial scanline is to be
      drawn by stepping one CPU instruction. Invoking this method will update the
      graphics buffer and generate the corresponding audio samples.
    */
    virtual void updateScanlineByStep() = 0;

    /**
      This method should be called whenever a new partial scanline is to be
      drawn by tracing to target address. Invoking this method will update the
      graphics buffer and generate the corresponding audio samples.
    */
    virtual void updateScanlineByTrace(int target) = 0;
#endif

  public:
    /**
      Answers the height of the frame buffer

      @return The frame's height
    */
    virtual uInt32 height() const = 0;

    /**
      Answers the width of the frame buffer

      @return The frame's width
    */
    virtual uInt32 width() const = 0;

  public:
    /**
      Answers the total number of scanlines the media source generated
      in producing the current frame buffer.

      @return The total number of scanlines generated
    */
    virtual uInt32 scanlines() const = 0;

    /**
      Sets the sound device for the TIA.
    */
    virtual void setSound(Sound& sound) = 0;

  private:
    // Copy constructor isn't supported by this class so make it private
    MediaSource(const MediaSource&);

    // Assignment operator isn't supported by this class so make it private
    MediaSource& operator = (const MediaSource&);
};

#endif
