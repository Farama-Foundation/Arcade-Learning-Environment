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
// $Id: Settings.cxx,v 1.125 2007/08/22 13:55:40 stephena Exp $
//============================================================================

#include <cassert>
#include <sstream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <string>

#include "ale/emucore/OSystem.hxx"
#include "ale/emucore/Settings.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Settings::Settings(OSystem* osystem) : myOSystem(osystem) {
    // Add this settings object to the OSystem
    myOSystem->attach(this);

    setInternal("palette", "standard");

    setInternal("sound", "false");
    setInternal("fragsize", "512");
    setInternal("freq", "31400");
    setInternal("tiafreq", "31400");
    setInternal("volume", "100");
    setInternal("clipvol", "true");

    setDefaultSettings();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Settings::~Settings()
{
  myInternalSettings.clear();
  myExternalSettings.clear();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::validate()
{
  std::string s;
  int i;

#ifdef SDL_SUPPORT
  i = getInt("volume");
  if(i < 0 || i > 100)
    setInternal("volume", "100");
  i = getInt("freq");
  if(i < 0 || i > 48000)
    setInternal("freq", "31400");
  i = getInt("tiafreq");
  if(i < 0 || i > 48000)
    setInternal("tiafreq", "31400");
#endif

  s = getString("palette");
  if(s != "standard" && s != "z26" && s != "user")
    setInternal("palette", "standard");
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setInt(const std::string& key, const int value)
{
  std::ostringstream stream;
  stream << value;

  if(int idx = getInternalPos(key) != -1){
    setInternal(key, stream.str(), idx);
  }
  else{
    verifyVariableExistence(intSettings, key);
    setExternal(key, stream.str());
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setFloat(const std::string& key, const float value)
{
  std::ostringstream stream;
  stream << value;

  if(int idx = getInternalPos(key) != -1){
    setInternal(key, stream.str(), idx);
  }
  else{
    verifyVariableExistence(floatSettings, key);
    setExternal(key, stream.str());
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setBool(const std::string& key, const bool value)
{
  std::ostringstream stream;
  stream << value;

  if(int idx = getInternalPos(key) != -1){
    setInternal(key, stream.str(), idx);
  }
  else{
    verifyVariableExistence(boolSettings, key);
    setExternal(key, stream.str());
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setString(const std::string& key, const std::string& value)
{
  if(int idx = getInternalPos(key) != -1){
    setInternal(key, value, idx);
  }
  else{
    verifyVariableExistence(stringSettings, key);
    setExternal(key, value);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::getSize(const std::string& key, int& x, int& y) const
{
  std::string size = getString(key);
  replace(size.begin(), size.end(), 'x', ' ');
  std::istringstream buf(size);
  buf >> x;
  buf >> y;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::getInt(const std::string& key, bool strict) const {
    // Try to find the named setting and answer its value
    int idx = -1;
    if((idx = getInternalPos(key)) != -1) {
        return (int) atoi(myInternalSettings[idx].value.c_str());
    } else {
        if((idx = getExternalPos(key)) != -1) {
            return (int) atoi(myExternalSettings[idx].value.c_str());
        } else {
            if (strict) {
                ale::Logger::Error << "No value found for key: " << key << ". ";
                ale::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                exit(-1);
            } else {
                return -1;
            }
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
float Settings::getFloat(const std::string& key, bool strict) const {
    // Try to find the named setting and answer its value
    int idx = -1;
    if((idx = getInternalPos(key)) != -1) {
        return (float) atof(myInternalSettings[idx].value.c_str());
    } else {
        if((idx = getExternalPos(key)) != -1) {
            return (float) atof(myExternalSettings[idx].value.c_str());
        } else {
            if (strict) {
                ale::Logger::Error << "No value found for key: " << key << ". ";
                ale::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                exit(-1);
            } else {
                return -1.0;
            }
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Settings::getBool(const std::string& key, bool strict) const {
    // Try to find the named setting and answer its value
    int idx = -1;
    if((idx = getInternalPos(key)) != -1)
    {
        const std::string& value = myInternalSettings[idx].value;
        if(value == "1" || value == "true" || value == "True")
            return true;
        else if(value == "0" || value == "false" || value == "False")
            return false;
        else
            return false;
    } else if((idx = getExternalPos(key)) != -1) {
        const std::string& value = myExternalSettings[idx].value;
        if(value == "1" || value == "true")
            return true;
        else if(value == "0" || value == "false")
            return false;
        else
            return false;
    } else {
        if (strict) {
            ale::Logger::Error << "No value found for key: " << key << ". ";
            ale::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
            exit(-1);
        } else {
            return false;
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const std::string& Settings::getString(const std::string& key, bool strict) const {
    // Try to find the named setting and answer its value
    int idx = -1;
    if((idx = getInternalPos(key)) != -1) {
        return myInternalSettings[idx].value;
    } else if ((idx = getExternalPos(key)) != -1) {
        return myExternalSettings[idx].value;
    } else {
        if (strict) {
            ale::Logger::Error << "No value found for key: " << key << ". ";
            ale::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
            exit(-1);
        } else {
            static std::string EmptyString("");
            return EmptyString;
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setSize(const std::string& key, const int value1, const int value2)
{
  std::ostringstream buf;
  buf << value1 << "x" << value2;
  setString(key, buf.str());
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::getInternalPos(const std::string& key) const
{
  for(unsigned int i = 0; i < myInternalSettings.size(); ++i)
    if(myInternalSettings[i].key == key)
      return i;

  return -1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::getExternalPos(const std::string& key) const
{
  for(unsigned int i = 0; i < myExternalSettings.size(); ++i)
    if(myExternalSettings[i].key == key)
      return i;

  return -1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::setInternal(const std::string& key, const std::string& value,
                          int pos, bool useAsInitial)
{
  int idx = -1;

  if(pos != -1 && pos >= 0 && pos < (int)myInternalSettings.size() &&
     myInternalSettings[pos].key == key)
  {
    idx = pos;
  }
  else
  {
    for(unsigned int i = 0; i < myInternalSettings.size(); ++i)
    {
      if(myInternalSettings[i].key == key)
      {
        idx = i;
        break;
      }
    }
  }

  if(idx != -1)
  {
    myInternalSettings[idx].key   = key;
    myInternalSettings[idx].value = value;
    if(useAsInitial) myInternalSettings[idx].initialValue = value;

    /*cerr << "modify internal: key = " << key
         << ", value  = " << value
         << ", ivalue = " << myInternalSettings[idx].initialValue
         << " @ index = " << idx
         << endl;*/
  }
  else
  {
    Setting setting;
    setting.key   = key;
    setting.value = value;
    if(useAsInitial) setting.initialValue = value;

    myInternalSettings.push_back(setting);
    idx = myInternalSettings.size() - 1;

    /*cerr << "insert internal: key = " << key
         << ", value  = " << value
         << ", ivalue = " << setting.initialValue
         << " @ index = " << idx
         << endl;*/
  }

  return idx;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::setExternal(const std::string& key, const std::string& value,
                          int pos, bool useAsInitial)
{
  int idx = -1;

  if(pos != -1 && pos >= 0 && pos < (int)myExternalSettings.size() &&
     myExternalSettings[pos].key == key)
  {
    idx = pos;
  }
  else
  {
    for(unsigned int i = 0; i < myExternalSettings.size(); ++i)
    {
      if(myExternalSettings[i].key == key)
      {
        idx = i;
        break;
      }
    }
  }

  if(idx != -1)
  {
    myExternalSettings[idx].key   = key;
    myExternalSettings[idx].value = value;
    if(useAsInitial) myExternalSettings[idx].initialValue = value;

    /*cerr << "modify external: key = " << key
         << ", value = " << value
         << " @ index = " << idx
         << endl;*/
  }
  else
  {
    Setting setting;
    setting.key   = key;
    setting.value = value;
    if(useAsInitial) setting.initialValue = value;

    myExternalSettings.push_back(setting);
    idx = myExternalSettings.size() - 1;

    /*cerr << "insert external: key = " << key
         << ", value = " << value
         << " @ index = " << idx
         << endl;*/
  }

  return idx;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Settings::Settings(const Settings&)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Settings& Settings::operator = (const Settings&)
{
  assert(false);

  return *this;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setDefaultSettings() {

    // Stella settings
    stringSettings.insert(std::pair<std::string, std::string>("cpu", "low")); // Reduce CPU emulation fidelity for speed
    // Random seed for ale::stella::System.
    // This random seed should be fixed to enable full determinism in the ALE
    intSettings.insert(std::pair<std::string, int>("system_random_seed", 4753849));

    // Controller settings
    intSettings.insert(std::pair<std::string, int>("max_num_frames", 0));
    intSettings.insert(std::pair<std::string, int>("max_num_frames_per_episode", 0));

    // Expose paddle_min and paddle_max settings but set as 'undefined' so
    // PADDLE_MIN and PADDLE_MAX defines are used as for default values in
    // the StellaEnvironment constructor.
    intSettings.insert(std::pair<std::string, int>("paddle_min", -1));
    intSettings.insert(std::pair<std::string, int>("paddle_max", -1));

    // FIFO controller settings
    boolSettings.insert(std::pair<std::string, bool>("run_length_encoding", true));

    // Environment customization settings
    boolSettings.insert(std::pair<std::string, bool>("restricted_action_set", false));
    intSettings.insert(std::pair<std::string, int>("random_seed", -1));
    boolSettings.insert(std::pair<std::string, bool>("color_averaging", false));
    boolSettings.insert(std::pair<std::string, bool>("send_rgb", false));
    intSettings.insert(std::pair<std::string, int>("frame_skip", 1));
    floatSettings.insert(std::pair<std::string, float>("repeat_action_probability", 0.25));
    stringSettings.insert(std::pair<std::string, std::string>("rom_file", ""));
    // Whether to truncate an episode on loss of life.
    boolSettings.insert(std::pair<std::string, bool>("truncate_on_loss_of_life", false));
    // Reward clipping settings
    intSettings.insert(std::pair<std::string, int>("reward_min", std::numeric_limits<int>::min()));
    intSettings.insert(std::pair<std::string, int>("reward_max", std::numeric_limits<int>::max()));

    // Record settings
    intSettings.insert(std::pair<std::string, int>("fragsize", 64)); // fragsize to 64 ensures proper sound sync
    stringSettings.insert(std::pair<std::string, std::string>("record_screen_dir", ""));
    stringSettings.insert(std::pair<std::string, std::string>("record_sound_filename", ""));

    // Display Settings
    boolSettings.insert(std::pair<std::string, bool>("display_screen", false));

    // Audio Settings
    boolSettings.insert(std::pair<std::string, bool>("sound_obs", false));

    for(std::map<std::string, std::string>::iterator it = stringSettings.begin(); it != stringSettings.end(); it++) {
      this->setString(it->first, it->second);
    }

    for(std::map<std::string, float>::iterator it = floatSettings.begin(); it != floatSettings.end(); it++) {
      this->setFloat(it->first, it->second);
    }

    for(std::map<std::string, bool>::iterator it = boolSettings.begin(); it != boolSettings.end(); it++) {
      this->setBool(it->first, it->second);
    }

    for(std::map<std::string, int>::iterator it = intSettings.begin(); it != intSettings.end(); it++) {
      this->setInt(it->first, it->second);
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template<typename ValueType>
void Settings::verifyVariableExistence(std::map<std::string, ValueType> dict, std::string key){
    if(dict.find(key) == dict.end()){
      throw std::runtime_error("The key " + key + " you are trying to set does not exist or has incorrect value type.\n");
    }
}

}  // namespace stella
}  // namespace ale
