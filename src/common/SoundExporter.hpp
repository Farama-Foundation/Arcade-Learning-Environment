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
        SoundExporter(const std::string &filename, int channels, bool record_for_user);
        ~SoundExporter();

        /** Adds a buffer of samples. */ 
        void addSamples(SampleType *s, int len);

        /** Clears buffer of samples since last user action. */ 
        void resetSamples();

        /** The sound data from beginning of episode. */
        std::vector<SampleType> m_data;

        /** Gets the latest audio data for user queries. */
        std::vector<SampleType> &getSamples();

        /** Flag indicating whether audio writing to file is enabled */
        bool m_record_to_file;

        /** Flag indicating whether audio buffer for user getAudio queries enabled */
        bool m_record_for_user;

    private:
   
        /** Writes the data to disk. */
        void writeWAVData();

        /** The file to save our audio to. */
        std::string m_filename;

        /** Number of channels. */
        int m_channels;

        /** Keep track of how many samples have been written since the last write to disk */
        size_t m_samples_since_write;
};

} // namespace ale::sound 
} // namespace ale

#endif // __SOUND_EXPORTER_HPP__ 
