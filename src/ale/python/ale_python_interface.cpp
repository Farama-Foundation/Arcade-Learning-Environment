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

#include "ale_python_interface.hpp"

namespace ale {

void ALEPythonInterface::getScreen(
    py::array_t<pixel_t, py::array::c_style>& buffer) {
  py::buffer_info info = buffer.request();
  if (info.ndim != 2) {
    throw std::runtime_error("Expected a numpy array with two dimensions.");
  }

  size_t h = environment->getScreen().height();
  size_t w = environment->getScreen().width();

  if (info.shape[0] != h || info.shape[1] != w) {
    std::stringstream msg;
    msg << "Invalid shape, (" << info.shape[0] << ", " << info.shape[1] << "), "
        << "expecting shape (" << h << ", " << w << ")";
    throw std::runtime_error(msg.str());
  }

  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = (pixel_t*)buffer.mutable_data();

  std::copy(src, src + (w * h * sizeof(pixel_t)), dst);
}

void ALEPythonInterface::getScreenRGB(
    py::array_t<pixel_t, py::array::c_style>& buffer) {
  py::buffer_info info = buffer.request();
  if (info.ndim != 3) {
    throw std::runtime_error("Expected a numpy array with three dimensions.");
  }

  size_t h = environment->getScreen().height();
  size_t w = environment->getScreen().width();

  if (info.shape[0] != h || info.shape[1] != w || info.shape[2] != 3) {
    std::stringstream msg;
    msg << "Invalid shape (" << info.shape[0] << ", " << info.shape[1] << ", "
        << info.shape[2] << "), expecting shape " << "(" << h << ", " << w << ", 3)";
    throw std::runtime_error(msg.str());
  }

  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = (pixel_t*)buffer.mutable_data();

  theOSystem->colourPalette().applyPaletteRGB(dst, src, w * h);
}

void ALEPythonInterface::getScreenGrayscale(
    py::array_t<pixel_t, py::array::c_style>& buffer) {
  py::buffer_info info = buffer.request();
  if (info.ndim != 2) {
    throw std::runtime_error("Expected a numpy array with two dimensions.");
  }

  size_t h = environment->getScreen().height();
  size_t w = environment->getScreen().width();

  if (info.shape[0] != h || info.shape[1] != w) {
    std::stringstream msg;
    msg << "Invalid shape (" << info.shape[0] << ", " << info.shape[1] << "), "
        << "expecting shape (" << h << ", " << w << ")";
    throw std::runtime_error(msg.str());
  }

  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = (pixel_t*)buffer.mutable_data();

  theOSystem->colourPalette().applyPaletteGrayscale(dst, src, h * w);
}

py::array_t<pixel_t, py::array::c_style> ALEPythonInterface::getScreen() {
  int32_t w = environment->getScreen().width();
  int32_t h = environment->getScreen().height();

  // Buffer info args:
  //   ptr: nullptr
  //   itemsize: sizeof(pixel_t)
  //   format: format_descriptor<pixel_t>::format()
  //   ndims: 2
  //   shape: { height, width }
  //   strides: { itemsize * w, itemsize } -- row major

   py::buffer_info info = py::buffer_info(
      nullptr, sizeof(pixel_t), py::format_descriptor<pixel_t>::format(), 2,
      {h, w}, {sizeof(pixel_t) * w, sizeof(pixel_t)});

  // Construct buffer with given info
  py::array_t<pixel_t, py::array::c_style> buffer(info);
  // Call our overloaded getScreen function
  this->getScreen(buffer);

  return buffer;
}

py::array_t<pixel_t, py::array::c_style> ALEPythonInterface::getScreenRGB() {
  int32_t h = environment->getScreen().height();
  int32_t w = environment->getScreen().width();

  // Buffer info args:
  //   ptr: nullptr
  //   itemsize: sizeof(pixel_t)
  //   format: format_descriptor<pixel_t>::format()
  //   ndims: 3
  //   shape: { height, width, 3 }
  //   strides: { itemsize * w * 3, itemsize * 3, itemsize } -- row major

   py::buffer_info info = py::buffer_info(
      nullptr, sizeof(pixel_t), py::format_descriptor<pixel_t>::format(), 3,
      {h, w, 3}, {sizeof(pixel_t) * w * 3, sizeof(pixel_t) * 3, sizeof(pixel_t)});

  // Construct buffer with given info
  py::array_t<pixel_t, py::array::c_style> buffer(info);
  // Call our overloaded getScreenRGB function
  this->getScreenRGB(buffer);

  return buffer;
}

py::array_t<pixel_t, py::array::c_style>
ALEPythonInterface::getScreenGrayscale() {
  int32_t w = environment->getScreen().width();
  int32_t h = environment->getScreen().height();

  // Buffer info args:
  //   ptr: nullptr
  //   itemsize: sizeof(pixel_t)
  //   format: format_descriptor<pixel_t>::format()
  //   ndims: 2
  //   shape: { height, width }
  //   strides: { itemsize * w, itemsize } -- row major

   py::buffer_info info = py::buffer_info(
      nullptr, sizeof(pixel_t), py::format_descriptor<pixel_t>::format(), 2,
      {h, w}, {sizeof(pixel_t) * w, sizeof(pixel_t)});

  // Construct buffer with given info
  py::array_t<pixel_t, py::array::c_style> buffer(info);
  // Call our overloaded getScreenGrayscale function
  this->getScreenGrayscale(buffer);

  return buffer;
}

const py::array_t<uint8_t, py::array::c_style> ALEPythonInterface::getAudio() {
  const std::vector<uint8_t> &audio = ALEInterface::getAudio();

  // Construct new py::array which copies audio data
  py::array_t<uint8_t, py::array::c_style> audio_array(audio.size(), audio.data());
  return audio_array;
}

void ALEPythonInterface::getAudio(
    py::array_t<uint8_t, py::array::c_style> &buffer) {
  py::buffer_info info = buffer.request();
  if (info.ndim != 1) {
    throw std::runtime_error("Expected a numpy array with one dimension.");
  }

  const std::vector<uint8_t> &audio = ALEInterface::getAudio();

  if (info.shape[0] != audio.size()) {
    std::stringstream msg;
    msg << "Invalid shape (" << info.shape[0] << "), "
        << "expecting shape (" << audio.size() << ")";
    throw std::runtime_error(msg.str());
  }

  // Get mutable data from buffer arg and copy audio data
  uint8_t *dst = (uint8_t *)buffer.mutable_data();
  std::copy(audio.data(), audio.data() + audio.size(), dst);
}

const py::array_t<uint8_t, py::array::c_style> ALEPythonInterface::getRAM() {
  const ALERAM& ram = ALEInterface::getRAM();

  // Construct new py::array which copies RAM
  py::array_t<uint8_t, py::array::c_style> ram_array(ram.size(), ram.array());
  return ram_array;
}

void ALEPythonInterface::getRAM(
    py::array_t<uint8_t, py::array::c_style>& buffer) {
  const ALERAM& ram = ALEInterface::getRAM();

  py::buffer_info info = buffer.request();
  if (info.ndim != 1) {
    throw std::runtime_error("Expected a numpy array with one dimension.");
  }

  if (info.shape[0] != ram.size()) {
    std::stringstream msg;
    msg << "Invalid shape (" << info.shape[0] << "), "
        << "expecting shape (" << ram.size() << ")";
    throw std::runtime_error(msg.str());
  }

  // Get mutable data from buffer arg and copy RAM
  pixel_t* dst = (pixel_t*)buffer.mutable_data();
  std::copy(ram.array(), ram.array() + ram.size(), dst);
}

} // namespace ale
