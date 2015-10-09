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
// $Id: Serializer.hxx,v 1.12 2007/01/01 18:04:49 stephena Exp $
//============================================================================

#ifndef SERIALIZER_HXX
#define SERIALIZER_HXX

#include <sstream>
#include "m6502/src/bspf/src/bspf.hxx"

/**
  This class implements a Serializer device, whereby data is
  serialized and sent to an output binary file in a system-
  independent way.

  All bytes and integers are written as int's.  Strings are
  written as characters prepended by the length of the string.
  Boolean values are written using a special pattern.

  @author  Stephen Anthony
  @version $Id: Serializer.hxx,v 1.12 2007/01/01 18:04:49 stephena Exp $
  
  Revised for ALE on Sep 20, 2009
  The new version uses a stringstream (not a file stream)
*/
class Serializer
{
  public:
    /**
      Creates a new Serializer device.

    */
    Serializer(void);

    /**
      Destructor
    */
    virtual ~Serializer(void);


    /**
      Closes the current output stream.
    */
    void close(void);
    
    bool isOpen(void) {return true;}

    /**
      Writes an int value to the current output stream.

      @param value The int value to write to the output stream.
    */
    void putInt(int value);

    /**
      Writes a string to the current output stream.

      @param str The string to write to the output stream.
    */
    void putString(const std::string& str);

    /**
      Writes a boolean value to the current output stream.

      @param b The boolean value to write to the output stream.
    */
    void putBool(bool b);

    // Accessor for myStream
    // TODO: don't copy the whole streams. 
    std::string get_str(void) const {
        return myStream.str();
    }
  private:
    // The stream to send the serialized data to.
    std::stringstream myStream;

    enum {
      TruePattern  = 0xfab1fab2,
      FalsePattern = 0xbad1bad2
    };
};

#endif
