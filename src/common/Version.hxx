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
// $Id: Version.hxx,v 1.29 2007/08/27 13:58:41 stephena Exp $
//============================================================================

#ifndef VERSION_HXX
#define VERSION_HXX

#define STELLA_BASE_VERSION "2.4.2"

#ifdef NIGHTLY_BUILD
#define STELLA_VERSION STELLA_BASE_VERSION "pre-" NIGHTLY_BUILD
#else
#define STELLA_VERSION STELLA_BASE_VERSION
#endif

#endif
