#include "SoundExporter.hpp"
#include <cassert>

namespace ale {
namespace sound {

// Sample rate is 60Hz x SamplesPerFrame bytes
// TODO(mgb): in reality this should be 31,400 Hz, but currently we are just short of this
static const unsigned int SampleRate = 60 * SoundExporter::SamplesPerFrame; 
// Save wav file every 30 seconds
static const unsigned int WriteInterval = SampleRate * 30;


SoundExporter::SoundExporter(const std::string &filename, int channels):
    m_filename(filename),
    m_channels(channels),
    m_samples_since_write(0) {
}


SoundExporter::~SoundExporter() {

    writeWAVData();
}


void SoundExporter::addSamples(SampleType *s, int len) {

    // @todo -- currently we only support mono recording 
    assert(m_channels == 1);

    for (int i = 0; i < len; i++)
        m_data.push_back(s[i]);

    // Periodically flush to disk (to avoid cases where the destructor is not called)
    m_samples_since_write += len;
    if (m_samples_since_write >= WriteInterval) {

        writeWAVData();
        m_samples_since_write = 0;
    }
}


void SoundExporter::writeWAVData() {
   
    // Taken from http://stackoverflow.com/questions/22226872/two-problems-when-writing-to-wav-c
    // Open file stream 
    std::ofstream stream(m_filename.c_str(), std::ios::binary);                

    // Cast size into a 32-bit integer
    int bufSize = m_data.size();

    // Header 
    stream.write("RIFF", 4);                                        // sGroupID (RIFF = Resource Interchange File Format)
    write<int>(stream, 36 + bufSize);                               // dwFileLength
    stream.write("WAVE", 4);                                        // sRiffType

    // Format chunk
    stream.write("fmt ", 4);                                        // sGroupID (fmt = format)
    write<int>(stream, 16);                                         // Chunk size (of Format Chunk)
    write<short>(stream, 1);                                        // Format (1 = PCM)
    write<short>(stream, m_channels);                                 // Channels
    write<int>(stream, SampleRate);                                 // Sample Rate
    write<int>(stream, SampleRate * m_channels * sizeof(SampleType)); // Byterate
    write<short>(stream, m_channels * sizeof(SampleType));            // Frame size aka Block align
    write<short>(stream, 8 * sizeof(SampleType));                   // Bits per sample

    // Data chunk
    stream.write("data", 4);                                        // sGroupID (data)
    stream.write((const char*)&bufSize, 4);                         // Chunk size (of Data, and thus of bufferSize)
    stream.write((const char*)&m_data[0], bufSize);                 // The samples DATA!!!
}

} // namespace ale::sound 
} // namespace ale

