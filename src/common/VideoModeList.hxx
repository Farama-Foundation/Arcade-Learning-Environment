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
// $Id: VideoModeList.hxx,v 1.3 2007/08/05 15:34:26 stephena Exp $
//============================================================================

#ifndef VIDMODE_LIST_HXX
#define VIDMODE_LIST_HXX

#include "Array.hxx"
#include "../emucore/m6502/src/bspf/src/bspf.hxx"

struct VideoMode {
  uInt32 image_x, image_y, image_w, image_h;
  uInt32 screen_w, screen_h;
  uInt32 zoom;
  string name;
};

/**
  This class implements an iterator around an array of VideoMode objects.

  @author  Stephen Anthony
  @version $Id: VideoModeList.hxx,v 1.3 2007/08/05 15:34:26 stephena Exp $
*/
class VideoModeList
{
  public:
    VideoModeList() : myIdx(-1) { }

    void add(VideoMode mode) { myModeList.push_back(mode); }

    void clear() { myModeList.clear(); }

    bool isEmpty() const { return myModeList.isEmpty(); }

    uInt32 size() const { return myModeList.size(); }

    const VideoMode& previous()
    {
      --myIdx;
      if(myIdx < 0) myIdx = myModeList.size() - 1;
      return current();
    }

    const VideoMode& current() const
    {
      return myModeList[myIdx];
    }

    const VideoMode& next()
    {
      myIdx = (myIdx + 1) % myModeList.size();
      return current();
    }

    void setByResolution(uInt32 width, uInt32 height)
    {
      // Find the largest resolution able to hold the given bounds
      myIdx = myModeList.size() - 1;
      for(unsigned int i = 0; i < myModeList.size(); ++i)
      {
        if(width <= myModeList[i].screen_w && height <= myModeList[i].screen_h)
        {
          myIdx = i;
          break;
        }
      }
    }

    void setByZoom(uInt32 zoom)
    {
      // Find the largest zoom within the given bounds
      myIdx = 0;
      for(unsigned int i = myModeList.size() - 1; i; --i)
      {
        if(myModeList[i].zoom <= zoom)
        {
          myIdx = i;
          break;
        }
      }
    }

  private:
    Common::Array<VideoMode> myModeList;
    int myIdx;
};

#endif
