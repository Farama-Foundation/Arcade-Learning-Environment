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
// $Id: OSystemUNIX.cxx,v 1.27 2007/07/19 16:21:39 stephena Exp $
//============================================================================

#include <cstdlib>
#include <sstream>
#include <fstream>

#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "bspf.hxx"
#include "OSystem.hxx"
#include "OSystemUNIX.hxx"

//ALE  #ifdef HAVE_GETTIMEOFDAY
#include <time.h>
#include <sys/time.h>
//ALE  #endif

/**
  Each derived class is responsible for calling the following methods
  in its constructor:

  setBaseDir()
  setConfigFile()

  See OSystem.hxx for a further explanation
*/

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
OSystemUNIX::OSystemUNIX()
  : OSystem()
{
  //ALE  const string& basedir = string(getenv("HOME")) + "/.stella";
  string basedir = string(".");  //ALE 
  setBaseDir(basedir);
  setConfigFile(basedir + "/stellarc");
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
OSystemUNIX::~OSystemUNIX()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 OSystemUNIX::getTicks() {
//ALE  #ifdef HAVE_GETTIMEOFDAY
    timeval now;
    gettimeofday(&now, 0);
    return (uInt32) (now.tv_sec * 1000000 + now.tv_usec);
//ALE  #else
//ALE    return (uInt32) SDL_GetTicks() * 1000;
//ALE  #endif
}
