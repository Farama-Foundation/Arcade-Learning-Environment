/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *   Matthew Hausknecht and the Reinforcement Learning and Artificial Intelligence
 *   Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  SoundRaw.cxx
 *
 *  A class for generating raw Atari 2600 sound samples.
 *
 **************************************************************************** */

#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Deserializer.hxx"

#include "ale/emucore/Settings.hxx"
#include "ale/common/SoundRaw.hxx"

#include "ale/common/Log.hpp"

namespace ale {
using namespace stella;   // Settings, Serializer, Deserializer

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SoundRaw::SoundRaw(Settings* settings)
  : Sound(settings),
    myIsEnabled(settings->getBool("sound_obs")),
    myIsInitializedFlag(false),
    myLastRegisterSetCycle(0)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SoundRaw::~SoundRaw()
{
  close();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundRaw::setEnabled(bool state)
{
  myIsEnabled = state;
  mySettings->setBool("sound_obs", state);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundRaw::initialize()
{
  // Check whether to start the sound subsystem
  if(!myIsEnabled)
  {
    close();
    return;
  }

  // Make sure the sound queue is clear
  myRegWriteQueue.clear();
  myTIASound.reset();

  myLastRegisterSetCycle = 0;
  myIsInitializedFlag = true;

  // Now initialize the TIASound object which will actually generate sound
  int frequency = mySettings->getInt("freq");
  myTIASound.outputFrequency(frequency);

  int tiafreq   = mySettings->getInt("tiafreq");
  myTIASound.tiaFrequency(tiafreq);

  // currently only support mono
  myTIASound.channels(1);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundRaw::close()
{
  myIsInitializedFlag = false;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool SoundRaw::isSuccessfullyInitialized() const
{
  return myIsInitializedFlag;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundRaw::reset()
{
  if(myIsInitializedFlag)
  {
    myLastRegisterSetCycle = 0;
    myTIASound.reset();
    myRegWriteQueue.clear();
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundRaw::adjustCycleCounter(int amount)
{
  myLastRegisterSetCycle += amount;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundRaw::set(uint16_t addr, uint8_t value, int cycle)
{
  TIARegister info;
  info.addr = addr;
  info.value = value;
  myRegWriteQueue.push_back(info);

  // Update last cycle counter to the current cycle
  myLastRegisterSetCycle = cycle;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void SoundRaw::process(uint8_t* buffer, uint32_t samples)
{
  // Process all the audio register updates up to this frame
  // Set audio registers
  uint32_t regSize = myRegWriteQueue.size();
  for(uint32_t i = 0; i < regSize; ++i) {
    TIARegister& info = myRegWriteQueue.front();
    myTIASound.set(info.addr, info.value);
    myRegWriteQueue.pop_front();
  }

  // Process audio registers
  myTIASound.process(buffer, samples);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool SoundRaw::load(Deserializer& in)
{
  std::string device = "TIASound";

  try
  {
    if(in.getString() != device)
      return false;

    uint8_t reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0, reg6 = 0;
    reg1 = (uint8_t) in.getInt();
    reg2 = (uint8_t) in.getInt();
    reg3 = (uint8_t) in.getInt();
    reg4 = (uint8_t) in.getInt();
    reg5 = (uint8_t) in.getInt();
    reg6 = (uint8_t) in.getInt();

    myLastRegisterSetCycle = (int) in.getInt();

    // Only update the TIA sound registers if sound is enabled
    // Make sure to empty the queue of previous sound fragments
    if(myIsInitializedFlag)
    {
      myRegWriteQueue.clear();
      myTIASound.set(0x15, reg1);
      myTIASound.set(0x16, reg2);
      myTIASound.set(0x17, reg3);
      myTIASound.set(0x18, reg4);
      myTIASound.set(0x19, reg5);
      myTIASound.set(0x1a, reg6);
    }
  }
  catch(char *msg)
  {
    ale::Logger::Error << msg << std::endl;
    return false;
  }
  catch(...)
  {
    ale::Logger::Error << "Unknown error in load state for " << device << std::endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool SoundRaw::save(Serializer& out)
{
  std::string device = "TIASound";

  try
  {
    out.putString(device);

    uint8_t reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0, reg6 = 0;

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
    ale::Logger::Error << msg << std::endl;
    return false;
  }
  catch(...)
  {
    ale::Logger::Error << "Unknown error in save state for " << device << std::endl;
    return false;
  }

  return true;
}

}  // namespace ale
