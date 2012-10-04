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
// $Id: EventStreamer.cxx,v 1.8 2007/01/01 18:04:48 stephena Exp $
//============================================================================

#include "bspf.hxx"

#include "OSystem.hxx"
#include "Event.hxx"
//ALE  #include "EventHandler.hxx"
#include "EventStreamer.hxx"
#include "System.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
EventStreamer::EventStreamer(OSystem* osystem)
  : myOSystem(osystem),
    myEventWriteFlag(false),
    myEventReadFlag(false),
    myFrameCounter(-1),
    myEventPos(0)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
EventStreamer::~EventStreamer()
{
  stopRecording();

  myEventHistory.clear();
  myStreamReader.close();
  myStreamWriter.close();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void EventStreamer::reset()
{
//cerr << "EventStreamer::reset()\n";
  myEventWriteFlag = false;
  myEventReadFlag = false;
  myFrameCounter = -1;
  myEventPos = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool EventStreamer::startRecording()
{
  string eventfile = /*myOSystem->baseDir() + BSPF_PATH_SEPARATOR +*/ "test.inp";
  if(!myStreamWriter.open(eventfile))
    return false;

  // And save the current state to it
  string md5 = myOSystem->console().properties().get(Cartridge_MD5);
  if(!myOSystem->console().system().saveState(md5, myStreamWriter))
    return false;
  myEventHistory.clear();

  reset();
  return myEventWriteFlag = true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool EventStreamer::stopRecording()
{
  if(!myStreamWriter.isOpen() || !myEventWriteFlag)
    return false;

  // Append the event history to the eventstream
  int size = myEventHistory.size();

  try
  {
    myStreamWriter.putString("EventStream");
    myStreamWriter.putInt(size);
    for(int i = 0; i < size; ++i)
      myStreamWriter.putInt(myEventHistory[i]);
  }
  catch(char *msg)
  {
    cerr << msg << endl;
    return false;
  }
  catch(...)
  {
    cerr << "Error saving eventstream" << endl;
    return false;
  }

  myStreamWriter.close();
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool EventStreamer::loadRecording()
{
cerr << "EventStreamer::loadRecording()\n";
  string eventfile = /*myOSystem->baseDir() + BSPF_PATH_SEPARATOR +*/ "test.inp";
  if(!myStreamReader.open(eventfile))
    return false;

  // Load ROM state
  string md5 = myOSystem->console().properties().get(Cartridge_MD5);
  if(!myOSystem->console().system().loadState(md5, myStreamReader))
    return false;

  try
  {
    if(myStreamReader.getString() != "EventStream")
      return false;

    // Now load the event stream
    myEventHistory.clear();
    int size = myStreamReader.getInt();
    for(int i = 0; i < size; ++i)
      myEventHistory.push_back(myStreamReader.getInt());
  }
  catch(char *msg)
  {
    cerr << msg << endl;
    return false;
  }
  catch(...)
  {
    cerr << "Error loading eventstream" << endl;
    return false;
  }

  reset();
  myEventReadFlag  = myEventHistory.size() > 0;

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void EventStreamer::addEvent(int type, int value)
{
  if(myEventWriteFlag)
  {
    myEventHistory.push_back(type);
    myEventHistory.push_back(value);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool EventStreamer::pollEvent(int& type, int& value)
{
  if(!myEventReadFlag)
    return false;

  bool status = false;

  // Read a new event from the stream when we've waited the appropriate
  // number of frames
  ++myFrameCounter;
  if(myFrameCounter >= 0)
  {
    int first = myEventHistory[myEventPos++];
    if(first < 0)
    {
      myFrameCounter = first;
      cerr << "wait " << -myFrameCounter << " frames\n";
    }
    else if(myEventPos < (int)myEventHistory.size())
    {
      type = first;
      value = myEventHistory[myEventPos++];
cerr << "type = " << type << ", value = " << value << endl;
      status = true;
    }
  }

  myEventReadFlag = myEventPos < (int)myEventHistory.size() - 2;
  return status;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void EventStreamer::nextFrame()
{
  if(myEventWriteFlag)
  {
    int idx = myEventHistory.size() - 1;
    if(idx >= 0 && myEventHistory[idx] < 0)
      --myEventHistory[idx];
    else
      myEventHistory.push_back(-1);
  }
}
