//============================================================================
//
//  BBBBB    SSSS   PPPPP   FFFFFF 
//  BB  BB  SS  SS  PP  PP  FF
//  BB  BB  SS      PP  PP  FF
//  BBBBB    SSSS   PPPPP   FFFF    --  "Brad's Simple Portability Framework"
//  BB  BB      SS  PP      FF
//  BB  BB  SS  SS  PP      FF
//  BBBBB    SSSS   PP      FF
//
// Copyright (c) 1997-1998 by Bradford W. Mott
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: bspf.hxx,v 1.17 2007/07/31 15:46:21 stephena Exp $
//============================================================================

#ifndef BSPF_HXX
#define BSPF_HXX

/**
  This file defines various basic data types and preprocessor variables
  that need to be defined for different operating systems.

  @author Bradford W. Mott
  @version $Id: bspf.hxx,v 1.17 2007/07/31 15:46:21 stephena Exp $
*/

// Types for 8-bit signed and unsigned integers
typedef signed char Int8;
typedef unsigned char uInt8;

// Types for 16-bit signed and unsigned integers
typedef signed short Int16;
typedef unsigned short uInt16;

// Types for 32-bit signed and unsigned integers
typedef signed int Int32;
typedef unsigned int uInt32;

// The following code should provide access to the standard C++ objects and
// types: cerr, cerr, string, ostream, istream, etc.
#ifdef BSPF_OLD_STYLE_CXX_HEADERS
  #include <iostream.h>
  #include <iomanip.h>
  #include <string.h>
#else
  #include <iostream>
  #include <iomanip>
  #include <string>
#endif
	
#ifdef HAVE_INTTYPES
  #include <inttypes.h>
#endif

// Defines to help with path handling
//ALE  #if defined BSPF_UNIX
#define BSPF_PATH_SEPARATOR  "/"
//ALE  #elif (defined(BSPF_DOS) || defined(BSPF_WIN32) || defined(BSPF_OS2))
//ALE    #define BSPF_PATH_SEPARATOR  "\\"
//ALE  #elif defined BSPF_MAC_OSX
//ALE    #define BSPF_PATH_SEPARATOR  "/"
//ALE  #elif defined BSPF_GP2X
//ALE      #define BSPF_PATH_SEPARATOR  "/"
//ALE  #endif

// I wish Windows had a complete POSIX layer
#ifdef BSPF_WIN32
  #define BSPF_strcasecmp stricmp
  #define BSPF_strncasecmp strnicmp
  #define BSPF_isblank(c) ((c == ' ') || (c == '\t'))
  #define BSPF_snprintf _snprintf
  #define BSPF_vsnprintf _vsnprintf
#else
  #define BSPF_strcasecmp strcasecmp
  #define BSPF_strncasecmp strncasecmp
  #define BSPF_isblank(c) isblank(c)
  #define BSPF_snprintf snprintf
  #define BSPF_vsnprintf vsnprintf
#endif

// Some convenience functions
template<typename T> inline void BSPF_swap(T &a, T &b) { T tmp = a; a = b; b = tmp; }
template<typename T> inline T BSPF_abs (T x) { return (x>=0) ? x : -x; }
template<typename T> inline T BSPF_min (T a, T b) { return (a<b) ? a : b; }
template<typename T> inline T BSPF_max (T a, T b) { return (a>b) ? a : b; }

#ifdef _WIN32_WCE
  #include "missing.h"
#endif

#endif
