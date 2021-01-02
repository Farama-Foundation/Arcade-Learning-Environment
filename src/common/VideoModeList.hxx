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

#include <vector>


struct VideoMode {
  uint32_t image_x, image_y, image_w, image_h;
  uint32_t screen_w, screen_h;
  uint32_t zoom;
  std::string name;
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

    uint32_t size() const { return myModeList.size(); }

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

    void setByResolution(uint32_t width, uint32_t height)
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

    void setByZoom(uint32_t zoom)
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
    std::vector<VideoMode> myModeList;
    int myIdx;
};

#endif
