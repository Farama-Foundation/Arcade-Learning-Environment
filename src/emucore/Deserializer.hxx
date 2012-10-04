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
// $Id: Deserializer.hxx,v 1.11 2007/01/01 18:04:47 stephena Exp $
//============================================================================

#ifndef DESERIALIZER_HXX
#define DESERIALIZER_HXX

#include <sstream>
#include "m6502/src/bspf/src/bspf.hxx"

/**
 This class implements a Deserializer device, whereby data is
 deserialized from an input binary file in a system-independent
 way.
 
 All ints should be cast to their appropriate data type upon method
 return.
 
 @author  Stephen Anthony
 @version $Id: Deserializer.hxx,v 1.11 2007/01/01 18:04:47 stephena Exp $
 
 Revised for ALE on Sep 20, 2009
 The new version uses a stringstream (not a file stream)
 
 TODO: don't copy the whole streams. 
 */
class Deserializer {
    public:
        /**
         Creates a new Deserializer device.
         */
        Deserializer(const string stream_str);
        
        void close(void);

        /**
         Reads an int value from the current input stream.
         
         @result The int value which has been read from the stream.
         */
        int getInt(void);
        
        /**
         Reads a string from the current input stream.
         
         @result The string which has been read from the stream.
         */
        string getString(void);
        
        /**
         Reads a boolean value from the current input stream.
         
         @result The boolean value which has been read from the stream.
         */
        bool getBool(void);
        
        bool isOpen(void) {return true;}
    private:
        // The stream to get the deserialized data from.
        stringstream myStream;
        
        enum {
            TruePattern  = 0xfab1fab2,
            FalsePattern = 0xbad1bad2
        };
    };

#endif
