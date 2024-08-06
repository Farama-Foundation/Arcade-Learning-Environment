//============================================================================
//
// MM     MM  6666  555555  0000   2222
// MMMM MMMM 66  66 55     00  00 22  22
// MM MMM MM 66     55     00  00     22
// MM  M  MM 66666  55555  00  00  22222  --  "A 6502 Microprocessor Emulator"
// MM     MM 66  66     55 00  00 22
// MM     MM 66  66 55  55 00  00 22
// MM     MM  6666   5555   0000  222222
//
// Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: System.cxx,v 1.21 2007/01/01 18:04:51 stephena Exp $
//============================================================================

#include <cassert>
#include <iostream>

#include "ale/emucore/Device.hxx"
#include "ale/emucore/M6502.hxx"
#include "ale/emucore/TIA.hxx"
#include "ale/emucore/System.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/Settings.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
System::System(Settings& settings)
  : myNumberOfDevices(0),
    myM6502(0),
    myTIA(0),
    myCycles(0),
    myDataBusState(0)
{
  // Seed RNG with fixed seed to enable full determinism
  int32_t emulatorSeed = settings.getInt("system_random_seed");
  myRandom.seed(emulatorSeed);

  // Allocate page table
  myPageAccessTable = new PageAccess[myNumberOfPages];

  // Initialize page access table
  PageAccess access;
  access.directPeekBase = 0;
  access.directPokeBase = 0;
  access.device = &myNullDevice;
  for(int page = 0; page < myNumberOfPages; ++page)
  {
    setPageAccess(page, access);
  }

  // Bus starts out unlocked (in other words, peek() changes myDataBusState)
  myDataBusLocked = false;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
System::~System()
{
  // Free the devices attached to me, since I own them
  for(uint32_t i = 0; i < myNumberOfDevices; ++i)
  {
    delete myDevices[i];
  }

  // Free the M6502 that I own
  delete myM6502;

  // Free my page access table
  delete[] myPageAccessTable;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void System::reset()
{
  // Reset system cycle counter
  resetCycles();

  // First we reset the devices attached to myself
  for(uint32_t i = 0; i < myNumberOfDevices; ++i)
  {
    myDevices[i]->reset();
  }

  // Now we reset the processor if it exists
  if(myM6502 != 0)
  {
    myM6502->reset();
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void System::attach(Device* device)
{
  assert(myNumberOfDevices < 100);

  // Add device to my collection of devices
  myDevices[myNumberOfDevices++] = device;

  // Ask the device to install itself
  device->install(*this);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void System::attach(M6502* m6502)
{
  // Remember the processor
  myM6502 = m6502;

  // Ask the processor to install itself
  myM6502->install(*this);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void System::attach(TIA* tia)
{
  myTIA = tia;
  attach((Device*) tia);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool System::save(Serializer& out)
{
  try
  {
    out.putString("System");
    out.putInt(myCycles);
    myRandom.saveState(out);
  }
  catch(char *msg)
  {
    std::cerr << msg << std::endl;
    return false;
  }
  catch(...)
  {
    std::cerr << "Unknown error in save state for \'System\'" << std::endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool System::load(Deserializer& in)
{
  try
  {
    if(in.getString() != "System")
      return false;

    myCycles = (uint32_t) in.getInt();
    myRandom.loadState(in);
  }
  catch(char *msg)
  {
    std::cerr << msg << std::endl;
    return false;
  }
  catch(...)
  {
    std::cerr << "Unknown error in load state for \'System\'" << std::endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void System::resetCycles()
{
  // First we let all of the device attached to me know about the reset
  for(uint32_t i = 0; i < myNumberOfDevices; ++i)
  {
    myDevices[i]->systemCyclesReset();
  }

  // Now, we reset cycle count to zero
  myCycles = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void System::setPageAccess(uint16_t page, const PageAccess& access)
{
  // Make sure the page is within range
  assert(page <= myNumberOfPages);

  // Make sure the access methods make sense
  assert(access.device != 0);

  myPageAccessTable[page] = access;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const System::PageAccess& System::getPageAccess(uint16_t page)
{
  // Make sure the page is within range
  assert(page <= myNumberOfPages);

  return myPageAccessTable[page];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool System::saveState(const std::string& md5sum, Serializer& out)
{
  // Open the file as a new Serializer
  if(!out.isOpen())
    return false;

  try
  {
    // Prepend the state file with the md5sum of this cartridge
    // This is the first defensive check for an invalid state file
    out.putString(md5sum);

    // First save state for this system
    if(!save(out))
      return false;

    // Next, save state for the CPU
    if(!myM6502->save(out))
      return false;

    // Now save the state of each device
    for(uint32_t i = 0; i < myNumberOfDevices; ++i)
      if(!myDevices[i]->save(out))
        return false;
  }
  catch(char *msg)
  {
    std::cerr << msg << std::endl;
    return false;
  }
  catch(...)
  {
    std::cerr << "Unknown error in save state for \'System\'" << std::endl;
    return false;
  }

  return true;  // success
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool System::loadState(const std::string& md5sum, Deserializer& in)
{
  // Open the file as a new Deserializer
  if(!in.isOpen())
    return false;

  try
  {
    // Look at the beginning of the state file.  It should contain the md5sum
    // of the current cartridge.  If it doesn't, this state file is invalid.
    if(in.getString() != md5sum)
      return false;

    // First load state for this system
    if(!load(in))
      return false;

    // Next, load state for the CPU
    if(!myM6502->load(in))
      return false;

    // Now load the state of each device
    for(uint32_t i = 0; i < myNumberOfDevices; ++i)
      if(!myDevices[i]->load(in))
        return false;
  }
  catch(char *msg)
  {
    std::cerr << msg << std::endl;
    return false;
  }
  catch(...)
  {
    std::cerr << "Unknown error in load state for \'System\'" << std::endl;
    return false;
  }

  return true;  // success
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
System::System(const System& s) {
  assert(false);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
System& System::operator = (const System&)
{
  assert(false);

  return *this;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void System::lockDataBus()
{
  myDataBusLocked = true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void System::unlockDataBus()
{
  myDataBusLocked = false;
}

}  // namespace stella
}  // namespace ale
