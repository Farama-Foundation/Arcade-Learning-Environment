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
// $Id: OSystemWin32.cxx,v 1.20 2007/08/04 20:32:54 stephena Exp $
//============================================================================

#include <sstream>
#include <fstream>
#include <windows.h>

#include "bspf.hxx"
#include "OSystem.hxx"
#include "OSystemWin32.hxx"

using namespace std;

/**
  Each derived class is responsible for calling the following methods
  in its constructor:

  setBaseDir()
  setConfigFile()

  See OSystem.hxx for a further explanation
*/

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
OSystemWin32::OSystemWin32()
  : OSystem()
{
  // TODO - there really should be code here to determine which version
  // of Windows is being used.
  // If using a version which supports multiple users (NT and above),
  // the relevant directories should be created in per-user locations.
  // For now, we just put it in the same directory as the executable.
  const string& basedir = ".";
  setBaseDir(basedir);
  setConfigFile(basedir + "\\stellarc");
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
OSystemWin32::~OSystemWin32()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 OSystemWin32::getTicks()
{
  //return (uInt32) SDL_GetTicks() * 1000;
    return static_cast<uInt32>(GetTickCount() * 1000);
}
