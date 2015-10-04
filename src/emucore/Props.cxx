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
// $Id: Props.cxx,v 1.20 2007/02/06 23:34:33 stephena Exp $
//============================================================================

#include <cctype>
#include <algorithm>
#include <sstream>
#include <string>
using namespace std;

#include "Props.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Properties::Properties()
{
  setDefaults();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Properties::Properties(const Properties& properties)
{
  copy(properties);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Properties::~Properties()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const string& Properties::get(PropertyType key) const
{
  if(key >= 0 && key < LastPropType)
    return myProperties[key];
  else {
    static std::string EmptyString("");
    return EmptyString;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Properties::set(PropertyType key, const string& value)
{
  if(key >= 0 && key < LastPropType)
  {
    myProperties[key] = value;

    switch(key)
    {
      case Cartridge_Sound:
      case Cartridge_Type:
      case Console_LeftDifficulty:
      case Console_RightDifficulty:
      case Console_TelevisionType:
      case Console_SwapPorts:
      case Controller_Left:
      case Controller_Right:
      case Controller_SwapPaddles:
      case Display_Format:
      case Display_Phosphor:
      case Emulation_HmoveBlanks:
      {
        transform(myProperties[key].begin(), myProperties[key].end(),
                  myProperties[key].begin(), (int(*)(int)) toupper);
        break;
      }

      case Display_PPBlend:
      {
        int blend = atoi(myProperties[key].c_str());
        if(blend < 0 || blend > 100) blend = 77;
        ostringstream buf;
        buf << blend;
        myProperties[key] = buf.str();
        break;
      }

      default:
        break;
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Properties::load(istream& in)
{
  setDefaults();

  string line, key, value;
  string::size_type one, two, three, four, garbage;

  // Loop reading properties
  while(getline(in, line))
  {
    // Strip all tabs from the line
    while((garbage = line.find("\t")) != string::npos)
      line.erase(garbage, 1);

    // Ignore commented and empty lines
    if((line.length() == 0) || (line[0] == ';'))
      continue;

    // End of this record
    if(line == "\"\"") 
      break;

    one = line.find("\"", 0);
    two = line.find("\"", one + 1);
    three = line.find("\"", two + 1);
    four = line.find("\"", three + 1);

    // Invalid line if it doesn't contain 4 quotes
    if((one == string::npos) || (two == string::npos) ||
       (three == string::npos) || (four == string::npos))
      break;

    // Otherwise get the key and value
    key = line.substr(one + 1, two - one - 1);
    value = line.substr(three + 1, four - three - 1);

    // Set the property 
    PropertyType type = getPropertyType(key);
    set(type, value);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Properties::save(ostream& out) const
{
  // Write out each of the key and value pairs
  bool changed = false;
  for(int i = 0; i < LastPropType; ++i)
  {
    // Try to save some space by only saving the items that differ from default
    if(myProperties[i] != ourDefaultProperties[i])
    {
      writeQuotedString(out, ourPropertyNames[i]);
      out.put(' ');
      writeQuotedString(out, myProperties[i]);
      out.put('\n');
      changed = true;
    }
  }

  if(changed)
  {
    // Put a trailing null string so we know when to stop reading
    writeQuotedString(out, "");
    out.put('\n');
    out.put('\n');
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
string Properties::readQuotedString(istream& in)
{
  char c;

  // Read characters until we see a quote
  while(in.get(c))
  {
    if(c == '"')
    {
      break;
    }
  }

  // Read characters until we see the close quote
  string s;
  while(in.get(c))
  {
    if((c == '\\') && (in.peek() == '"'))
    {
      in.get(c);
    }
    else if((c == '\\') && (in.peek() == '\\'))
    {
      in.get(c);
    }
    else if(c == '"')
    {
      break;
    }
    else if(c == '\r')
    {
      continue;
    }

    s += c;
  }

  return s;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Properties::writeQuotedString(ostream& out, const string& s)
{
  out.put('"');
  for(uInt32 i = 0; i < s.length(); ++i)
  {
    if(s[i] == '\\')
    {
      out.put('\\');
      out.put('\\');
    }
    else if(s[i] == '\"')
    {
      out.put('\\');
      out.put('"');
    }
    else
    {
      out.put(s[i]);
    }
  }
  out.put('"');
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Properties& Properties::operator = (const Properties& properties)
{
  // Do the assignment only if this isn't a self assignment
  if(this != &properties)
  {
    // Now, make myself a copy of the given object
    copy(properties);
  }

  return *this;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Properties::copy(const Properties& properties)
{
  // Now, copy each property from properties
  for(int i = 0; i < LastPropType; ++i)
    myProperties[i] = properties.myProperties[i];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Properties::print() const
{
  cerr << get(Cartridge_MD5) << "|"
       << get(Cartridge_Name) << "|"
       << get(Cartridge_Rarity) << "|"
       << get(Cartridge_Manufacturer) << "|"
       << get(Cartridge_Note)
       << endl;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Properties::setDefaults()
{
  for(int i = 0; i < LastPropType; ++i)
    myProperties[i] = ourDefaultProperties[i];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PropertyType Properties::getPropertyType(const string& name)
{
  for(int i = 0; i < LastPropType; ++i)
    if(ourPropertyNames[i] == name)
      return (PropertyType)i;

  // Otherwise, indicate that the item wasn't found
  return LastPropType;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* Properties::ourDefaultProperties[LastPropType] = {
  "",            // Cartridge.MD5
  "",            // Cartridge.Manufacturer
  "",            // Cartridge.ModelNo
  "Untitled",    // Cartridge.Name
  "",            // Cartridge.Note
  "",            // Cartridge.Rarity
  "MONO",        // Cartridge.Sound
  "AUTO-DETECT", // Cartridge.Type
  "B",           // Console.LeftDifficulty
  "B",           // Console.RightDifficulty
  "COLOR",       // Console.TelevisionType
  "NO",          // Console.SwapPorts
  "JOYSTICK",    // Controller.Left
  "JOYSTICK",    // Controller.Right
  "NO",          // Controller.SwapPaddles
  "AUTO-DETECT", // Display.Format
  "34",          // Display.YStart
  "210",         // Display.Height
  "NO",          // Display.Phosphor
  "77",          // Display.PPBlend
  "YES"          // Emulation.HmoveBlanks
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* Properties::ourPropertyNames[LastPropType] = {
  "Cartridge.MD5",
  "Cartridge.Manufacturer",
  "Cartridge.ModelNo",
  "Cartridge.Name",
  "Cartridge.Note",
  "Cartridge.Rarity",
  "Cartridge.Sound",
  "Cartridge.Type",
  "Console.LeftDifficulty",
  "Console.RightDifficulty",
  "Console.TelevisionType",
  "Console.SwapPorts",
  "Controller.Left",
  "Controller.Right",
  "Controller.SwapPaddles",
  "Display.Format",
  "Display.YStart",
  "Display.Height",
  "Display.Phosphor",
  "Display.PPBlend",
  "Emulation.HmoveBlanks"
};
