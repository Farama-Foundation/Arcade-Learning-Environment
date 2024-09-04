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


namespace ale {
namespace stella {

class MediaSource;
class Sound;

}  // namespace stella
}  // namespace ale

#include <cstdint>

namespace ale {
namespace stella {

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
    virtual uint8_t* currentFrameBuffer() const = 0;

    /**
      Answers the previous frame buffer

      @return Pointer to the previous frame buffer
    */
    virtual uint8_t* previousFrameBuffer() const = 0;

  public:
    /**
      Answers the height of the frame buffer

      @return The frame's height
    */
    virtual uint32_t height() const = 0;

    /**
      Answers the width of the frame buffer

      @return The frame's width
    */
    virtual uint32_t width() const = 0;

  public:
    /**
      Answers the total number of scanlines the media source generated
      in producing the current frame buffer.

      @return The total number of scanlines generated
    */
    virtual uint32_t scanlines() const = 0;

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

}  // namespace stella
}  // namespace ale

#endif
