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
// $Id: Random.cxx,v 1.4 2007/01/01 18:04:49 stephena Exp $
//============================================================================

#include <time.h>
#include "Random.hxx"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Random::seed(uInt32 value)
{
  ourSeed = value;
  ourSeeded = true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Random::Random()
{
  // If we haven't been seeded then seed ourself
  if(!ourSeeded)
  {
    cerr << "random seed " << endl;
    ourSeed = (uInt32)time(0);
    ourSeeded = true;
  }

  myValue = ourSeed;
}
 
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 Random::next()
{
  return (myValue = (myValue * 2416 + 374441) % 1771875);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uInt32 Random::ourSeed = 0;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Random::ourSeeded = false;

