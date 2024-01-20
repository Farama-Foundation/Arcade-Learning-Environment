/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  ale_python_interface.cpp
 *
 *  Bindings for the ALE Python Interface.
 *
 **************************************************************************** */

#include <array>

#include "ale_python_interface.hpp"

namespace ale {

void ALEPythonInterface::getScreenRGB(
    py::array_t<uint8_t, py::array::c_style>& buffer) {
  py::buffer_info info = buffer.request();
  if (info.ndim != 3) {
    throw std::runtime_error("Expected a numpy array with three dimensions.");
  }

  stella::MediaSource& media = ALEInterface::theOSystem->console().mediaSource();
  int32_t h = media.height();
  int32_t w = media.width();

  if (info.shape[0] != h || info.shape[1] != w || info.shape[2] != 3) {
    std::stringstream msg;
    msg << "Invalid shape (" << info.shape[0] << ", " << info.shape[1] << ", "
        << info.shape[2] << "), expecting shape " << "(" << h << ", " << w << ", 3)";
    throw std::runtime_error(msg.str());
  }

  uint8_t* dst = (uint8_t*)buffer.mutable_data();
  ALEInterface::frameProcessor->processRGB(media, dst);
}

void ALEPythonInterface::getScreenGrayscale(
    py::array_t<uint8_t, py::array::c_style>& buffer) {
  py::buffer_info info = buffer.request();
  if (info.ndim != 2) {
    throw std::runtime_error("Expected a numpy array with two dimensions.");
  }

  stella::MediaSource& media = ALEInterface::theOSystem->console().mediaSource();
  int32_t h = media.height();
  int32_t w = media.width();

  if (info.shape[0] != h || info.shape[1] != w) {
    std::stringstream msg;
    msg << "Invalid shape (" << info.shape[0] << ", " << info.shape[1] << "), "
        << "expecting shape (" << h << ", " << w << ")";
    throw std::runtime_error(msg.str());
  }

  uint8_t* dst = (uint8_t*)buffer.mutable_data();
  ALEInterface::frameProcessor->processGrayscale(media, dst);
}

py::array_t<uint8_t, py::array::c_style> ALEPythonInterface::getScreenRGB() {
  stella::MediaSource& media = ALEInterface::theOSystem->console().mediaSource();
  int32_t h = media.height();
  int32_t w = media.width();

  // Buffer info args:
  //   ptr: nullptr
  //   itemsize: sizeof(uint8_t)
  //   format: format_descriptor<uint8_t>::format()
  //   ndims: 3
  //   shape: { height, width, 3 }
  //   strides: { itemsize * w * 3, itemsize * 3, itemsize } -- row major

   py::buffer_info info = py::buffer_info(
      nullptr, sizeof(uint8_t), py::format_descriptor<uint8_t>::format(), 3,
      {h, w, 3}, {sizeof(uint8_t) * w * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)});

  // Construct buffer with given info
  py::array_t<uint8_t, py::array::c_style> buffer(info);
  // Call our overloaded getScreenRGB function
  this->getScreenRGB(buffer);

  return buffer;
}

py::array_t<uint8_t, py::array::c_style>
ALEPythonInterface::getScreenGrayscale() {
  stella::MediaSource& media = ALEInterface::theOSystem->console().mediaSource();
  int32_t h = media.height();
  int32_t w = media.width();

  // Buffer info args:
  //   ptr: nullptr
  //   itemsize: sizeof(uint8_t)
  //   format: format_descriptor<uint8_t>::format()
  //   ndims: 2
  //   shape: { height, width }
  //   strides: { itemsize * w, itemsize } -- row major

   py::buffer_info info = py::buffer_info(
      nullptr, sizeof(uint8_t), py::format_descriptor<uint8_t>::format(), 2,
      {h, w}, {sizeof(uint8_t) * w, sizeof(uint8_t)});

  // Construct buffer with given info
  py::array_t<uint8_t, py::array::c_style> buffer(info);
  // Call our overloaded getScreenGrayscale function
  this->getScreenGrayscale(buffer);

  return buffer;
}

const py::array_t<uint8_t, py::array::c_style> ALEPythonInterface::getRAM() {
  // Construct new py::array which copies RAM
  py::array_t<uint8_t, py::array::c_style> ram_array(this->getRAMSize());
  this->getRAM(ram_array);
  return ram_array;
}

void ALEPythonInterface::getRAM(
    py::array_t<uint8_t, py::array::c_style>& buffer) {
  py::buffer_info info = buffer.request();
  if (info.ndim != 1) {
    throw std::runtime_error("Expected a numpy array with one dimension.");
  }

  if (info.shape[0] != this->getRAMSize()) {
    std::stringstream msg;
    msg << "Invalid shape (" << info.shape[0] << "), "
        << "expecting shape (" << this->getRAMSize() << ")";
    throw std::runtime_error(msg.str());
  }

  // Get mutable data from buffer arg and copy RAM
  uint8_t* dst = (uint8_t*)buffer.mutable_data();
  for (size_t i = 0; i < 128; ++i) {
    dst[i] = ALEInterface::theOSystem->console().system().peek(i + 0x80);
  }
}

} // namespace ale
