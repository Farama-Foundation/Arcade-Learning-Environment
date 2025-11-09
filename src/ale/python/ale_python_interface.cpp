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

// Screen methods with dynamic shapes (varies by game)
void ALEPythonInterface::getScreen(
    nb::ndarray<pixel_t, nb::c_contig, nb::device::cpu>& buffer) {
  if (buffer.ndim() != 2) {
    throw std::runtime_error("Expected a numpy array with two dimensions.");
  }

  size_t h = environment->getScreen().height();
  size_t w = environment->getScreen().width();

  if (buffer.shape(0) != h || buffer.shape(1) != w) {
    std::stringstream msg;
    msg << "Invalid shape, (" << buffer.shape(0) << ", " << buffer.shape(1) << "), "
        << "expecting shape (" << h << ", " << w << ")";
    throw std::runtime_error(msg.str());
  }

  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = buffer.data();

  std::copy(src, src + (w * h), dst);
}

void ALEPythonInterface::getScreenRGB(
    nb::ndarray<pixel_t, nb::c_contig, nb::device::cpu>& buffer) {
  if (buffer.ndim() != 3) {
    throw std::runtime_error("Expected a numpy array with three dimensions.");
  }

  size_t h = environment->getScreen().height();
  size_t w = environment->getScreen().width();

  if (buffer.shape(0) != h || buffer.shape(1) != w || buffer.shape(2) != 3) {
    std::stringstream msg;
    msg << "Invalid shape (" << buffer.shape(0) << ", " << buffer.shape(1) << ", "
        << buffer.shape(2) << "), expecting shape " << "(" << h << ", " << w << ", 3)";
    throw std::runtime_error(msg.str());
  }

  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = buffer.data();

  theOSystem->colourPalette().applyPaletteRGB(dst, src, w * h);
}

void ALEPythonInterface::getScreenGrayscale(
    nb::ndarray<pixel_t, nb::c_contig, nb::device::cpu>& buffer) {
  if (buffer.ndim() != 2) {
    throw std::runtime_error("Expected a numpy array with two dimensions.");
  }

  size_t h = environment->getScreen().height();
  size_t w = environment->getScreen().width();

  if (buffer.shape(0) != h || buffer.shape(1) != w) {
    std::stringstream msg;
    msg << "Invalid shape (" << buffer.shape(0) << ", " << buffer.shape(1) << "), "
        << "expecting shape (" << h << ", " << w << ")";
    throw std::runtime_error(msg.str());
  }

  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = buffer.data();

  theOSystem->colourPalette().applyPaletteGrayscale(dst, src, h * w);
}

nb::ndarray<nb::numpy, pixel_t> ALEPythonInterface::getScreen() {
  size_t w = environment->getScreen().width();
  size_t h = environment->getScreen().height();

  // Create a new numpy array with dynamic shape
  size_t shape[2] = {h, w};
  auto result = nb::ndarray<nb::numpy, pixel_t>(
      nullptr, 2, shape);

  // Copy data from the screen buffer
  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = result.data();
  std::copy(src, src + (h * w), dst);

  return result;
}

nb::ndarray<nb::numpy, pixel_t> ALEPythonInterface::getScreenRGB() {
  size_t h = environment->getScreen().height();
  size_t w = environment->getScreen().width();

  // Create a new numpy array with dynamic shape (h, w, 3)
  size_t shape[3] = {h, w, 3};
  auto result = nb::ndarray<nb::numpy, pixel_t>(
      nullptr, 3, shape);

  // Apply RGB palette
  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = result.data();
  theOSystem->colourPalette().applyPaletteRGB(dst, src, w * h);

  return result;
}

nb::ndarray<nb::numpy, pixel_t>
ALEPythonInterface::getScreenGrayscale() {
  size_t w = environment->getScreen().width();
  size_t h = environment->getScreen().height();

  // Create a new numpy array with dynamic shape
  size_t shape[2] = {h, w};
  auto result = nb::ndarray<nb::numpy, pixel_t>(
      nullptr, 2, shape);

  // Apply grayscale palette
  pixel_t* src = environment->getScreen().getArray();
  pixel_t* dst = result.data();
  theOSystem->colourPalette().applyPaletteGrayscale(dst, src, h * w);

  return result;
}

// Audio methods with static shape (512,)
nb::ndarray<nb::numpy, uint8_t, nb::shape<512>> ALEPythonInterface::getAudio() {
  const std::vector<uint8_t> &audio = ALEInterface::getAudio();

  // Create a new numpy array with static shape (512,)
  size_t shape[1] = {512};
  auto result = nb::ndarray<nb::numpy, uint8_t, nb::shape<512>>(
      nullptr, 1, shape);
  std::copy(audio.data(), audio.data() + audio.size(), result.data());

  return result;
}

void ALEPythonInterface::getAudio(
    nb::ndarray<uint8_t, nb::shape<512>, nb::c_contig, nb::device::cpu> &buffer) {
  if (buffer.ndim() != 1) {
    throw std::runtime_error("Expected a numpy array with one dimension.");
  }

  const std::vector<uint8_t> &audio = ALEInterface::getAudio();

  if (buffer.shape(0) != 512) {
    std::stringstream msg;
    msg << "Invalid shape (" << buffer.shape(0) << "), "
        << "expecting shape (512)";
    throw std::runtime_error(msg.str());
  }

  // Get mutable data from buffer arg and copy audio data
  uint8_t *dst = buffer.data();
  std::copy(audio.data(), audio.data() + audio.size(), dst);
}

// RAM methods with static shape (128,)
nb::ndarray<nb::numpy, uint8_t, nb::shape<128>> ALEPythonInterface::getRAM() {
  const ALERAM& ram = ALEInterface::getRAM();

  // Create a new numpy array with static shape (128,)
  size_t shape[1] = {128};
  auto result = nb::ndarray<nb::numpy, uint8_t, nb::shape<128>>(
      nullptr, 1, shape);
  std::copy(ram.array(), ram.array() + ram.size(), result.data());

  return result;
}

void ALEPythonInterface::getRAM(
    nb::ndarray<uint8_t, nb::shape<128>, nb::c_contig, nb::device::cpu>& buffer) {
  const ALERAM& ram = ALEInterface::getRAM();

  if (buffer.ndim() != 1) {
    throw std::runtime_error("Expected a numpy array with one dimension.");
  }

  if (buffer.shape(0) != 128) {
    std::stringstream msg;
    msg << "Invalid shape (" << buffer.shape(0) << "), "
        << "expecting shape (128)";
    throw std::runtime_error(msg.str());
  }

  // Get mutable data from buffer arg and copy RAM
  uint8_t* dst = buffer.data();
  std::copy(ram.array(), ram.array() + ram.size(), dst);
}

} // namespace ale