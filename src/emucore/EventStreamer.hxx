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
// $Id: EventStreamer.hxx,v 1.5 2007/01/01 18:04:48 stephena Exp $
//============================================================================

#ifndef EVENTSTREAMER_HXX
#define EVENTSTREAMER_HXX

#include "../common/Array.hxx"
#include "Deserializer.hxx"
#include "Serializer.hxx"

class OSystem;

/**
  This class takes care of event streams, which consist of a ROM state
  file and appended event data.  This appended data is defined as a
  string of integers which are grouped as follows:

  event; event ; ... ; -framewait; event ... ; -framewait; ...

  'event' consists of type/value pairs (each event in Stella is an
  enumerated type with an associated value)
  'framewait' is the number of frames to wait until executing all
  the following events

  The EventStreamer can load and save eventstream recordings.  When in 
  'save' mode, all events are queued from the Event class, and appended to
  the ROM state when recording was started.

  When in 'load' mode, the ROM state is loaded from the eventstream, and
  the appended event data is available in a queue for polling.
  It's the responsibility of the calling object (most likely the EventHandler)
  to poll for data; this class simply makes the queued events available in
  the correct order at the correct time.

  @author  Stephen Anthony
  @version $Id: EventStreamer.hxx,v 1.5 2007/01/01 18:04:48 stephena Exp $
*/
class EventStreamer
{
  public:
    /**
      Create a new event streamer object
    */
    EventStreamer(OSystem* osystem);
 
    /**
      Destructor
    */
    virtual ~EventStreamer();

  public:
    /**
      Start recording event-stream to disk
    */
    bool startRecording();

    /**
      Stop recording event-stream
    */
    bool stopRecording();

    /**
      Load recorded event-stream into the system
    */
    bool loadRecording();

    /**
      Adds the given event to the event history
    */
    void addEvent(int type, int value);

    /**
      Gets the next event from the event history
    */
    bool pollEvent(int& type, int& value);

    /**
      Answers if we're in recording mode
    */
    bool isRecording() { return myEventWriteFlag; }

    /**
      Indicate that a new frame has been processed
    */
    void nextFrame();

    /**
      Reset to base state (not saving or loading an eventstream)
    */
    void reset();

  private:

  private:
    // Global OSystem object
    OSystem* myOSystem;

    // Indicates if we're in save/write or load/read mode
    bool myEventWriteFlag;
    bool myEventReadFlag;

    // Current frame count (used for waiting while polling)
    int myFrameCounter;

    // Current position in the event history queue
    int myEventPos;

    // Stores the history/record of all events that have been set
    IntArray myEventHistory;

    // Serializer classes used to save/load the eventstream
    Serializer   myStreamWriter;
    Deserializer myStreamReader;
};

#endif
