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
// $Id: Serializer.cxx,v 1.11 2007/01/01 18:04:49 stephena Exp $
//============================================================================

#include "Serializer.hxx"
using namespace std;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Serializer::Serializer(void) {
    myStream.clear();
    
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Serializer::~Serializer(void)
{
    close();
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Serializer::close(void)
{
    myStream.clear();
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Serializer::putInt(int value)
{
    unsigned char buf[4];
    for(int i = 0; i < 4; ++i)
        buf[i] = (value >> (i<<3)) & 0xff;
    
    myStream.write((char*)buf, 4);
    if(myStream.bad())
        throw "Serializer: file write failed";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Serializer::putString(const string& str)
{
    int len = str.length();
    putInt(len);
    myStream.write(str.data(), (streamsize)len);
    
    if(myStream.bad())
        throw "Serializer: file write failed";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void Serializer::putBool(bool b)
{
    putInt(b ? TruePattern: FalsePattern);
}

