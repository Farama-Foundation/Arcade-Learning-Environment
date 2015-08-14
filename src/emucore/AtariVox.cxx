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
// $Id: AtariVox.cxx,v 1.5 2007/01/01 18:04:44 stephena Exp $
//============================================================================

#ifdef ATARIVOX_SUPPORT

#include "Event.hxx"
#include "AtariVox.hxx"
#include "SpeakJet.hxx"
#include "../common/Log.hpp"

#define DEBUG_ATARIVOX 0

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
AtariVox::AtariVox(Jack jack, const Event& event)
    : Controller(jack, event),
      mySpeakJet(0),
      mySystem(0),
      myPinState(0),
      myShiftCount(0),
      myShiftRegister(0),
      myLastDataWriteCycle(0)
{
  myType = Controller::AtariVox;
  mySpeakJet = new SpeakJet();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
AtariVox::~AtariVox()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void AtariVox::setSystem(System *system) {
  mySystem = system;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool AtariVox::read(DigitalPin pin)
{
  // For now, always return true, meaning the device is ready
/*
  if(DEBUG_ATARIVOX)
    cerr << "AtariVox: read from SWCHA" << endl;
*/
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Int32 AtariVox::read(AnalogPin)
{
  // Analog pins are not connected in AtariVox, so we have infinite resistance 
  return maximumResistance;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void AtariVox::clockDataIn(bool value)
{
  // bool oldValue = myPinState & 0x01;
  myPinState = (myPinState & 0xfe) | value;

  uInt32 cycle = mySystem->cycles();
  if(DEBUG_ATARIVOX)
    ale::Logger::Info << "AtariVox: value "
                      << value
                      << " written to DATA line at "
                      << mySystem->cycles()
                      << " (-"
                      << myLastDataWriteCycle
                      << "=="
                      << (mySystem->cycles() - myLastDataWriteCycle)
                      << ")"
                      << endl;

  if(value && (myShiftCount == 0)) {
    if(DEBUG_ATARIVOX)
      ale::Logger::Info << "value && (myShiftCount == 0), returning" << endl;
    return;
  }

  if(cycle < myLastDataWriteCycle || cycle > myLastDataWriteCycle + 1000) {
    // If this is the first write this frame, or if it's been a long time
    // since the last write, start a new data byte.
    myShiftRegister = 0;
    myShiftCount = 0;
  }

  if(cycle < myLastDataWriteCycle || cycle >= myLastDataWriteCycle + 62) {
    // If this is the first write this frame, or if it's been 62 cycles
    // since the last write, shift this bit into the current byte.
    if(DEBUG_ATARIVOX)
      ale::Logger::Info << "cycle >= myLastDataWriteCycle + 62, shiftIn("
           << value << ")" << endl;
    shiftIn(value);
  }

  myLastDataWriteCycle = cycle;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void AtariVox::shiftIn(bool value)
{
  myShiftRegister >>= 1;
  myShiftRegister |= (value << 15);
  if(++myShiftCount == 10) {
    myShiftCount = 0;
    myShiftRegister >>= 6;
    if(!(myShiftRegister & (1<<9)))
      ale::Logger::Warning << "AtariVox: bad start bit" << endl;
    else if((myShiftRegister & 1))
      ale::Logger::Warning << "AtariVox: bad stop bit" << endl;
    else
    {
      uInt8 data = ((myShiftRegister >> 1) & 0xff);
      ale::Logger::Warning << "AtariVox: output byte " << ((int)(data)) << endl;
      mySpeakJet->write(data);
    }
    myShiftRegister = 0;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void AtariVox::write(DigitalPin pin, bool value)
{
  if(DEBUG_ATARIVOX)
    ale::Logger::Info << "AtariVox: write to SWCHA" << endl;

  // Change the pin state based on value
  switch(pin)
  {
    // Pin 1 is the DATA line, used to output serial data to the
    // speakjet
    case One:
        clockDataIn(value);
      break;
  
    // Pin 2 is the SDA line, used to output data to the 24LC256
    // serial EEPROM, using the I2C protocol.
    // I'm not even trying to emulate this right now :(
    case Two:
      if(DEBUG_ATARIVOX)
        ale::Logger::Info << "AtariVox: value "
                          << value
                          << " written to SDA line at cycle "
                          << mySystem->cycles()
                          << endl;
      break;

    // Pin 2 is the SCLK line, used to output clock data to the 24LC256
    // serial EEPROM, using the I2C protocol.
    // I'm not even trying to emulate this right now :(
    case Three:
      if(DEBUG_ATARIVOX)
        ale::Logger::Info << "AtariVox: value "
                          << value
                          << " written to SCLK line at cycle "
                          << mySystem->cycles()
                          << endl;
      break;

    case Four:
    default:
      break;
  } 
}

#endif
