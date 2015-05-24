/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *   Matthew Hausknecht and the Reinforcement Learning and Artificial Intelligence 
 *   Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  SoundExporter.hpp 
 *
 *  A class for writing Atari 2600 sound to a WAV file.
 *
 *  Parts of this code were taken from 
 *
 *  http://stackoverflow.com/questions/22226872/two-problems-when-writing-to-wav-c
 *
 **************************************************************************** */

#ifndef __SOUND_EXPORTER_HPP__
#define __SOUND_EXPORTER_HPP__ 

#include <fstream>
#include <vector>
#include "../emucore/m6502/src/bspf/src/bspf.hxx"

namespace ale {
namespace sound {

template <typename T>
void write(std::ofstream& stream, const T& t) {
    stream.write((const char*)&t, sizeof(T));
}

class SoundExporter {

    public:
    
        static const int SamplesPerFrame = 512;

        typedef uInt8 SampleType;
  
        /** Create a new sound exporter which, on program termination, will write out a wav file. */
        SoundExporter(const std::string &filename, int channels);
        ~SoundExporter();

        /** Adds a buffer of samples. */ 
        void addSamples(SampleType *s, int len);

    private:
   
        /** Writes the data to disk. */
        void writeWAVData();

        /** The file to save our audio to. */
        std::string m_filename;

        /** Number of channels. */
        int m_channels;

        /** The sound data. */
        std::vector<SampleType> m_data;

        /** Keep track of how many samples have been written since the last write to disk */
        size_t m_samples_since_write;
};

} // namespace ale::sound 
} // namespace ale

#endif // __SOUND_EXPORTER_HPP__ 
