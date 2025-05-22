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
    nb::ndarray<nb::numpy, pixel_t, nb::c_contig> buffer) {
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

  std::copy(src, src + (w * h * sizeof(pixel_t)), dst);
}

void ALEPythonInterface::getScreenRGB(
    nb::ndarray<nb::numpy, pixel_t, nb::c_contig> buffer) {
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
    nb::ndarray<nb::numpy, pixel_t, nb::c_contig> buffer) {
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
  int32_t w = environment->getScreen().width();
  int32_t h = environment->getScreen().height();

  // Create new array with shape {h, w}
  size_t shape[2] = {static_cast<size_t>(h), static_cast<size_t>(w)};
  auto buffer = nb::steal(nb::detail::ndarray_new(
      nb::handle(nb::dtype<pixel_t>().raw_dtype()),
      2, shape, nb::handle(nullptr), nullptr, nb::dtype<pixel_t>()));

  // Call our overloaded getScreen function
  this->getScreen(nb::cast<nb::ndarray<nb::numpy, pixel_t, nb::c_contig>>(buffer));

  return nb::cast<nb::ndarray<nb::numpy, pixel_t>>(buffer);
}

nb::ndarray<nb::numpy, pixel_t> ALEPythonInterface::getScreenRGB() {
  int32_t h = environment->getScreen().height();
  int32_t w = environment->getScreen().width();

  // Create new array with shape {h, w, 3}
  size_t shape[3] = {static_cast<size_t>(h), static_cast<size_t>(w), 3};
  auto buffer = nb::steal(nb::detail::ndarray_new(
      nb::handle(nb::dtype<pixel_t>().raw_dtype()),
      3, shape, nb::handle(nullptr), nullptr, nb::dtype<pixel_t>()));

  // Call our overloaded getScreenRGB function
  this->getScreenRGB(nb::cast<nb::ndarray<nb::numpy, pixel_t, nb::c_contig>>(buffer));

  return nb::cast<nb::ndarray<nb::numpy, pixel_t>>(buffer);
}

nb::ndarray<nb::numpy, pixel_t>
ALEPythonInterface::getScreenGrayscale() {
  int32_t w = environment->getScreen().width();
  int32_t h = environment->getScreen().height();

  // Create new array with shape {h, w}
  size_t shape[2] = {static_cast<size_t>(h), static_cast<size_t>(w)};
  auto buffer = nb::steal(nb::detail::ndarray_new(
      nb::handle(nb::dtype<pixel_t>().raw_dtype()),
      2, shape, nb::handle(nullptr), nullptr, nb::dtype<pixel_t>()));

  // Call our overloaded getScreenGrayscale function
  this->getScreenGrayscale(nb::cast<nb::ndarray<nb::numpy, pixel_t, nb::c_contig>>(buffer));

  return nb::cast<nb::ndarray<nb::numpy, pixel_t>>(buffer);
}

nb::ndarray<nb::numpy, uint8_t> ALEPythonInterface::getAudio() {
  const std::vector<uint8_t> &audio = ALEInterface::getAudio();

  // Create new array and copy data
  size_t shape[1] = {audio.size()};
  auto buffer = nb::steal(nb::detail::ndarray_new(
      nb::handle(nb::dtype<uint8_t>().raw_dtype()),
      1, shape, nb::handle(nullptr), nullptr, nb::dtype<uint8_t>()));

  auto array = nb::cast<nb::ndarray<nb::numpy, uint8_t, nb::c_contig>>(buffer);
  std::copy(audio.data(), audio.data() + audio.size(), array.data());

  return nb::cast<nb::ndarray<nb::numpy, uint8_t>>(buffer);
}

void ALEPythonInterface::getAudio(
    nb::ndarray<nb::numpy, uint8_t, nb::c_contig> buffer) {
  if (buffer.ndim() != 1) {
    throw std::runtime_error("Expected a numpy array with one dimension.");
  }

  const std::vector<uint8_t> &audio = ALEInterface::getAudio();

  if (buffer.shape(0) != audio.size()) {
    std::stringstream msg;
    msg << "Invalid shape (" << buffer.shape(0) << "), "
        << "expecting shape (" << audio.size() << ")";
    throw std::runtime_error(msg.str());
  }

  // Get mutable data from buffer arg and copy audio data
  uint8_t *dst = buffer.data();
  std::copy(audio.data(), audio.data() + audio.size(), dst);
}

nb::ndarray<nb::numpy, uint8_t> ALEPythonInterface::getRAM() {
  const ALERAM& ram = ALEInterface::getRAM();

  // Create new array and copy data
  size_t shape[1] = {static_cast<size_t>(ram.size())};
  auto buffer = nb::steal(nb::detail::ndarray_new(
      nb::handle(nb::dtype<uint8_t>().raw_dtype()),
      1, shape, nb::handle(nullptr), nullptr, nb::dtype<uint8_t>()));

  auto array = nb::cast<nb::ndarray<nb::numpy, uint8_t, nb::c_contig>>(buffer);
  std::copy(ram.array(), ram.array() + ram.size(), array.data());

  return nb::cast<nb::ndarray<nb::numpy, uint8_t>>(buffer);
}

void ALEPythonInterface::getRAM(
    nb::ndarray<nb::numpy, uint8_t, nb::c_contig> buffer) {
  const ALERAM& ram = ALEInterface::getRAM();

  if (buffer.ndim() != 1) {
    throw std::runtime_error("Expected a numpy array with one dimension.");
  }

  if (buffer.shape(0) != ram.size()) {
    std::stringstream msg;
    msg << "Invalid shape (" << buffer.shape(0) << "), "
        << "expecting shape (" << ram.size() << ")";
    throw std::runtime_error(msg.str());
  }

  // Get mutable data from buffer arg and copy RAM
  uint8_t* dst = buffer.data();
  std::copy(ram.array(), ram.array() + ram.size(), dst);
}

} // namespace ale
