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
#include <algorithm>

#include "OSystem.hxx"
#include "Version.hxx"
#include "bspf.hxx"
#include "Settings.hxx"
#include "GuiUtils.hxx"  //ALE 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Settings::Settings(OSystem* osystem) : myOSystem(osystem) {
    // Add this settings object to the OSystem
    myOSystem->attach(this);

    // Add options that are common to all versions of Stella
    setInternal("video", "soft");

    setInternal("gl_filter", "nearest");
    setInternal("gl_aspect", "100");
    setInternal("gl_fsmax", "never");
    setInternal("gl_lib", "libGL.so");
    setInternal("gl_vsync", "false");
    setInternal("gl_texrect", "false");

    setInternal("zoom_ui", "2");
    setInternal("zoom_tia", "2");
    setInternal("fullscreen", "false");
    setInternal("fullres", "");
    setInternal("center", "true");
    setInternal("grabmouse", "false");
    setInternal("palette", "standard");
    setInternal("colorloss", "false");

    setInternal("sound", "false");
    setInternal("fragsize", "512");
    setInternal("freq", "31400");
    setInternal("tiafreq", "31400");
    setInternal("volume", "100");
    setInternal("clipvol", "true");

    setInternal("keymap", "");
    setInternal("joymap", "");
    setInternal("joyaxismap", "");
    setInternal("joyhatmap", "");
    setInternal("paddle", "0");
    setInternal("sa1", "left");
    setInternal("sa2", "right");
    setInternal("p0speed", "50");
    setInternal("p1speed", "50");
    setInternal("p2speed", "50");
    setInternal("p3speed", "50");
    setInternal("pthresh", "600");

    setInternal("showinfo", "false");

    setInternal("ssdir", string(".") + BSPF_PATH_SEPARATOR);
    setInternal("sssingle", "false");

    setInternal("romdir", "");
    setInternal("statedir", "");
    setInternal("cheatfile", "");
    setInternal("palettefile", "");
    setInternal("propsfile", "");
    setInternal("working_dir",  string(".") + BSPF_PATH_SEPARATOR);
    setInternal("rl_params_file",  "rl_params.txt");
    setInternal("class_disc_params_file",  "class_disc_params.txt");
    setInternal("rombrowse", "true");
    setInternal("lastrom", "");

    setInternal("debuggerres", "1030x690");
    setInternal("launcherres", "400x300");
    setInternal("uipalette", "0");
    setInternal("mwheel", "4");
    setInternal("autoslot", "false");
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Settings::~Settings()
{
  myInternalSettings.clear();
  myExternalSettings.clear();
}

void Settings::loadConfig(const char* config_file){
    string line, key, value;
    string::size_type equalPos, garbage;

    ifstream in(config_file);
    if(!in || !in.is_open()) {
    cerr << "Warning: couldn't load settings file: " << config_file << "\n";
    return;
    }

    while(getline(in, line)) {
        // Strip all whitespace and tabs from the line
        while((garbage = line.find("\t")) != string::npos)
          line.erase(garbage, 1);

        // Ignore commented and empty lines
        if((line.length() == 0) || (line[0] == ';'))
          continue;

        // Search for the equal sign and discard the line if its not found
        if((equalPos = line.find("=")) == string::npos)
          continue;

        // Split the line into key/value pairs and trim any whitespace
        key   = line.substr(0, equalPos);
        value = line.substr(equalPos + 1, line.length() - key.length() - 1);
        key   = trim(key);
        value = trim(value);

        // Check for absent key or value
        if((key.length() == 0) || (value.length() == 0))
          continue;

        // Only settings which have been previously set are valid
        //ALE  if(int idx = getInternalPos(key) != -1)
        //ALE  setInternal(key, value, idx, true);
        setInternal(key, value);
    }

    in.close();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::loadConfig()
{
 loadConfig(myOSystem->configFile().c_str());
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
string Settings::loadCommandLine(int argc, char** argv)
{
  for(int i = 1; i < argc; ++i)
  {
    // strip off the '-' character
    string key = argv[i];
    if(key[0] == '-')
    {
      key = key.substr(1, key.length());

      // Take care of the arguments which are meant to be executed immediately
      // (and then Stella should exit)
      if(key == "help" || key == "listrominfo")
      {
        usage();
        setExternal(key, "true");
        return "";
      }

      // Take care of arguments without an option
      if(key == "rominfo" || key == "debug" || key == "holdreset" ||
         key == "holdselect" || key == "holdbutton0")
      {
        setExternal(key, "true");
        continue;
      }

      if(++i >= argc)
      {
        cerr << "Missing argument for '" << key << "'" << endl;
        return "";
      }
      string value = argv[i];

      // Settings read from the commandline must not be saved to 
      // the rc-file, unless they were previously set
      if(int idx = getInternalPos(key) != -1)
        setInternal(key, value, idx);   // don't set initialValue here
      else
        setExternal(key, value);
    }
    else
      return key;
  }

  return "";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::validate()
{
  string s;
  int i;

  s = getString("video");
  if(s != "soft" && s != "gl")
    setInternal("video", "soft");

#ifdef DISPLAY_OPENGL
  s = getString("gl_filter");
  if(s != "linear" && s != "nearest")
    setInternal("gl_filter", "nearest");

  i = getInt("gl_aspect");
  if(i < 50 || i > 100)
    setInternal("gl_aspect", "100");

  s = getString("gl_fsmax");
  if(s != "never" && s != "ui" && s != "tia" && s != "always")
    setInternal("gl_fsmax", "never");
#endif

#ifdef SOUND_SUPPORT
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

  i = getInt("zoom_ui");
  if(i < 1 || i > 10)
    setInternal("zoom_ui", "2");

  i = getInt("zoom_tia");
  if(i < 1 || i > 10)
    setInternal("zoom_tia", "2");

  i = getInt("paddle");
  if(i < 0 || i > 3)
    setInternal("paddle", "0");

  i = getInt("pthresh");
  if(i < 400)
    setInternal("pthresh", "400");
  else if(i > 800)
    setInternal("pthresh", "800");

  s = getString("palette");
  if(s != "standard" && s != "z26" && s != "user")
    setInternal("palette", "standard");
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::usage() {

    cerr << "\n" 
       " ***************************************************************************\n"         
       " * Welcome to A.L.E (Arcade Learning Environment)\n"                                
       " * (Powered by Stella)\n"
       " ***************************************************************************\n"       
       "\n"
       " Usage: ale [options ...] romfile"
       "\n"
       " Example Usage: ./ale -player_agent random_agent roms/asterix.bin\n" 
       "\n"
       " Main arguments:\n"
       "   -help -- prints out help information\n\n"
       "   -game_controller [internal|fifo|fifo_named"
#ifdef __USE_RLGLUE
       "|rlglue"
#endif
       "]\n"
       "      Defines how Stella communicates with the player agent:\n"                  
       "            - 'internal':   (default) an instance of the PlayerAgent\n"       
       "                            subclass controls the game\n"   
       "            - 'fifo':       Control occurs through FIFO pipes\n"
       "            - 'fifo_named': Control occurs through named FIFO pipes\n"
#ifdef __USE_RLGLUE
       "            - 'rlglue':     External control via RL-Glue\n"
#endif
       "\n"
       "   -random_seed  [time]/[n]\n"
       "     Sets the seed used for random number generation.\n"
       "     'time' will use the the current time.\n"
       "\n"
#ifdef __USE_SDL
       "   -display_screen [true]/[false]\n"
       "     Displays the game screen.\n"
       "\n"
#endif
       "   -output_file [filename] -- if true and SDL is enabled, displays ALE screen\n"
       "     default: not set\n\n"
       "\n"
       " Environment arguments:\n"
       "   -max_num_episodes n\n"
       "     The program will quit after this number of episodes. 0 means never.\n"
       "    default: 0\n"
       "   -max_num_frames m\n"  
       "     The program will quit after this number of frames. 0 means never.\n"
       "    default: 0\n"
       "   -max_num_frames_per_episode m\n"
       "     Ends each episode after this number of frames. 0 means never.\n"
       "    default: 0\n"
       "   -system_reset_steps ### -- how many frames to hold reset button for\n" 
       "      Should be >= 2.\n"
       "    default: 4\n\n"
       "   -use_environment_distribution [true|false]  -- if true, the environment start\n" 
       "      state is drawn from a distribution of states\n"
       "    default: false\n\n"
       "   -use_starting_actions [true|false] -- if true, a game-specific sequence\n"
       "      of actions is applied after each reset\n"
       "    default: false\n\n"
       "   -restricted_action_set [true|false] -- if true, agents use a smaller set of\n" 
       "      actions (internal interface only)\n"
       "    default: false\n\n"
       "   -backward_compatible_save [true|false] -- if true, uses ALE 0.2's\n" 
       "      save/load state mechanism\n"
       "    default: false\n\n"
       "   -disable_color_averaging [true|false] -- if true, disables color averaging\n" 
       "    default: false\n\n"
       "\n"
       " FIFO arguments:\n"
       "   -run_length_encoding [true|false] -- if true, encodes data using run-length encoding\n"
       "    default: true\n\n"
       "\n"
       " Internal Controller arguments:\n"
       "   -player_agent [random_agent|single_action_agent"
#ifdef __USE_SDL
       "|keyboard_agent" 
#endif
       "]\n"                                                                   
       "     Specifies which PlayerAgent to run from the internal controller.\n\n"
       "   -record_trajectory <true|false> -- if true, records the agent's trajectory\n"
       "    default: false\n\n"
       "   -agent_epsilon ### -- probability of a random action in single_action_agent\n"
       "    default: unset\n\n"
       "\n"
       " Misc. arguments:\n"
       "   -ld [A/B]\n"
       "     Left player difficulty. B (default) means easy.\n"
       "\n"
       "   -rd [A/B]\n"
       "     Right player difficulty. B (default) means easy.\n"
       "\n"
    ; // Closing the std::cerr statement
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::saveConfig()
{
  // Do a quick scan of the internal settings to see if any have
  // changed.  If not, we don't need to save them at all.
  bool settingsChanged = false;
  for(unsigned int i = 0; i < myInternalSettings.size(); ++i)
  {
    if(myInternalSettings[i].value != myInternalSettings[i].initialValue)
    {
      settingsChanged = true;
      break;
    }
  }

  if(!settingsChanged)
    return;

  ofstream out(myOSystem->configFile().c_str());
  if(!out || !out.is_open())
  {
    cerr << "Error: Couldn't save settings file\n";
    return;
  }

  out << ";  Stella configuration file" << endl
      << ";" << endl
      << ";  Lines starting with ';' are comments and are ignored." << endl
      << ";  Spaces and tabs are ignored." << endl
      << ";" << endl
      << ";  Format MUST be as follows:" << endl
      << ";    command = value" << endl
      << ";" << endl
      << ";  Commmands are the same as those specified on the commandline," << endl
      << ";  without the '-' character." << endl
      << ";" << endl
      << ";  Values are the same as those allowed on the commandline." << endl
      << ";  Boolean values are specified as 1 (or true) and 0 (or false)" << endl
      << ";" << endl;

  // Write out each of the key and value pairs
  for(unsigned int i = 0; i < myInternalSettings.size(); ++i)
  {
    out << myInternalSettings[i].key << " = " <<
           myInternalSettings[i].value << endl;
  }

  out.close();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setInt(const string& key, const int value)
{
  ostringstream stream;
  stream << value;

  if(int idx = getInternalPos(key) != -1)
    setInternal(key, stream.str(), idx);
  else
    setExternal(key, stream.str());
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setFloat(const string& key, const float value)
{
  ostringstream stream;
  stream << value;

  if(int idx = getInternalPos(key) != -1)
    setInternal(key, stream.str(), idx);
  else
    setExternal(key, stream.str());
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setBool(const string& key, const bool value)
{
  ostringstream stream;
  stream << value;

  if(int idx = getInternalPos(key) != -1)
    setInternal(key, stream.str(), idx);
  else
    setExternal(key, stream.str());
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setString(const string& key, const string& value)
{
  if(int idx = getInternalPos(key) != -1)
    setInternal(key, value, idx);
  else
    setExternal(key, value);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::getSize(const string& key, int& x, int& y) const
{
  string size = getString(key);
  replace(size.begin(), size.end(), 'x', ' ');
  istringstream buf(size);
  buf >> x;
  buf >> y;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::getInt(const string& key, bool strict) const {
    // Try to find the named setting and answer its value
    int idx = -1;
    if((idx = getInternalPos(key)) != -1) {
        return (int) atoi(myInternalSettings[idx].value.c_str());
    } else { 
        if((idx = getExternalPos(key)) != -1) {
            return (int) atoi(myExternalSettings[idx].value.c_str());
        } else {
            if (strict) {
                cerr << "No value found for key: " << key << ". ";
                cerr << "Make sure all the settings files are loaded." << endl;
                exit(-1);
            } else {
                return -1;
            }
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
float Settings::getFloat(const string& key, bool strict) const {
    // Try to find the named setting and answer its value
    int idx = -1;
    if((idx = getInternalPos(key)) != -1) {
        return (float) atof(myInternalSettings[idx].value.c_str());
    } else { 
        if((idx = getExternalPos(key)) != -1) {
            return (float) atof(myExternalSettings[idx].value.c_str());
        } else {
            if (strict) {
                cerr << "No value found for key: " << key << ". ";
                cerr << "Make sure all the settings files are loaded." << endl;
                exit(-1);
            } else {
                return -1.0;
            }
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Settings::getBool(const string& key, bool strict) const {
    // Try to find the named setting and answer its value
    int idx = -1;
    if((idx = getInternalPos(key)) != -1)
    {
        const string& value = myInternalSettings[idx].value;
        if(value == "1" || value == "true" || value == "True")
            return true;
        else if(value == "0" || value == "false" || value == "False")
            return false;
        else
            return false;
    } else if((idx = getExternalPos(key)) != -1) {
        const string& value = myExternalSettings[idx].value;
        if(value == "1" || value == "true")
            return true;
        else if(value == "0" || value == "false")
            return false;
        else
            return false;
    } else {
        if (strict) {
            cerr << "No value found for key: " << key << ". ";
            cerr << "Make sure all the settings files are loaded." << endl;
            exit(-1);
        } else {
            return false;
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const string& Settings::getString(const string& key, bool strict) const {
    // Try to find the named setting and answer its value
    int idx = -1;
    if((idx = getInternalPos(key)) != -1) {
        return myInternalSettings[idx].value;
    } else if ((idx = getExternalPos(key)) != -1) {
        return myExternalSettings[idx].value;
    } else {
        if (strict) {
            cerr << "No value found for key: " << key << ". ";
            cerr << "Make sure all the settings files are loaded." << endl;
            exit(-1);
        } else {
            return EmptyString;
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Settings::setSize(const string& key, const int value1, const int value2)
{
  ostringstream buf;
  buf << value1 << "x" << value2;
  setString(key, buf.str());
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::getInternalPos(const string& key) const
{
  for(unsigned int i = 0; i < myInternalSettings.size(); ++i)
    if(myInternalSettings[i].key == key)
      return i;

  return -1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::getExternalPos(const string& key) const
{
  for(unsigned int i = 0; i < myExternalSettings.size(); ++i)
    if(myExternalSettings[i].key == key)
      return i;

  return -1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int Settings::setInternal(const string& key, const string& value,
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
int Settings::setExternal(const string& key, const string& value,
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
