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
// Copyright (c) 1995-2007 by Bradford W. Mott
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: TIASnd.hxx,v 1.6 2007/01/01 18:04:50 stephena Exp $
//============================================================================

#ifndef TIASOUND_HXX
#define TIASOUND_HXX
#include <cstdint>

namespace ale {
namespace stella {

/**
  This class implements a fairly accurate emulation of the TIA sound
  hardware.

  @author  Bradford W. Mott
  @version $Id: TIASnd.hxx,v 1.6 2007/01/01 18:04:50 stephena Exp $
*/
class TIASound
{
  public:
    /**
      Create a new TIA Sound object using the specified output frequency
    */
    TIASound(int outputFrequency = 31400, int tiaFrequency = 31400,
             uint32_t channels = 1);

    /**
      Destructor
    */
    virtual ~TIASound();

  public:
    /**
      Reset the sound emulation to its power-on state
    */
    void reset();

    /**
      Set the frequency output samples should be generated at
    */
    void outputFrequency(int freq);

    /**
      Set the frequency the of the TIA device
    */
    void tiaFrequency(int freq);

    /**
      Selects the number of audio channels per sample (1 = mono, 2 = stereo)
    */
    void channels(uint32_t number);

    /**
      Set volume clipping (decrease volume range by half to eliminate popping)
    */
    void clipVolume(bool clip);

  public:
    /**
      Sets the specified sound register to the given value

      @param address Register address
      @param value Value to store in the register
    */
    void set(uint16_t address, uint8_t value);

    /**
      Gets the specified sound register's value

      @param address Register address
    */
    uint8_t get(uint16_t address);

    /**
      Create sound samples based on the current sound register settings
      in the specified buffer. NOTE: If channels is set to stereo then
      the buffer will need to be twice as long as the number of samples.

      @param buffer The location to store generated samples
      @param samples The number of samples to generate
    */
    void process(uint8_t* buffer, uint32_t samples);

    /**
      Set the volume of the samples created (0-100)
    */
    void volume(uint32_t percent);

  private:
    /**
      Frequency divider class which outputs 1 after "divide-by" clocks. This
      is used to divide the main frequency by the values 1 to 32.
    */
    class FreqDiv
    {
      public:
        FreqDiv()
        {
          myDivideByValue = myCounter = 0;
        }

        void set(uint32_t divideBy)
        {
          myDivideByValue = divideBy;
        }

        bool clock()
        {
          if(++myCounter > myDivideByValue)
          {
            myCounter = 0;
            return true;
          }
          return false;
        }

      private:
        uint32_t myDivideByValue;
        uint32_t myCounter;
    };

  private:
    uint8_t myAUDC[2];
    uint8_t myAUDF[2];
    uint8_t myAUDV[2];

    FreqDiv myFreqDiv[2];    // Frequency dividers
    uint8_t myP4[2];           // 4-bit register LFSR (lower 4 bits used)
    uint8_t myP5[2];           // 5-bit register LFSR (lower 5 bits used)

    int  myOutputFrequency;
    int  myTIAFrequency;
    uint32_t myChannels;
    int  myOutputCounter;
    uint32_t myVolumePercentage;
    uint8_t  myVolumeClip;
};

}  // namespace stella
}  // namespace ale

#endif
