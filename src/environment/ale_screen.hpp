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
 *  ale_screen.hpp
 *
 *  A class that encapsulates an Atari 2600 screen. Code is provided inline for
 *   efficiency reasonss.
 *  
 **************************************************************************** */

#ifndef __ALE_SCREEN_HPP__
#define __ALE_SCREEN_HPP__

#include <string.h>
#include <memory>
#include <vector>

typedef unsigned char pixel_t;

/** A simple wrapper around an Atari screen. */ 
class ALEScreen { 
  public:
    ALEScreen(int h, int w);
    ALEScreen(const ALEScreen &rhs);

    ALEScreen& operator=(const ALEScreen &rhs);

    /** pixel accessors, (row, column)-ordered */
    pixel_t get(int r, int c) const;
    pixel_t *pixel(int r, int c);
    
    /** Access a whole row */
    pixel_t *getRow(int r) const;
    
    /** Access the whole array */
    pixel_t *getArray() const { return const_cast<pixel_t *>(&m_pixels[0]); }

    /** Dimensionality information */
    size_t height() const { return m_rows; }
    size_t width() const { return m_columns; }

    /** Returns the size of the underlying array */
    size_t arraySize() const { return m_rows * m_columns * sizeof(pixel_t); }

    /** Returns whether two screens are equal */
    bool equals(const ALEScreen &rhs) const;

  protected:
    int m_rows;
    int m_columns;

    std::vector<pixel_t> m_pixels; 
};

inline ALEScreen::ALEScreen(int h, int w):
  m_rows(h),
  m_columns(w),
  // Create a pixel array of the requisite size
  m_pixels(m_rows * m_columns) {
}

inline ALEScreen::ALEScreen(const ALEScreen &rhs):
  m_rows(rhs.m_rows),
  m_columns(rhs.m_columns),
  m_pixels(rhs.m_pixels) {

}

inline ALEScreen& ALEScreen::operator=(const ALEScreen &rhs) {
  
  m_rows = rhs.m_rows;
  m_columns = rhs.m_columns;

  // We rely here on the std::vector constructor doing something sensible (i.e. not wasteful)
  // inside its assignment operator
  m_pixels = rhs.m_pixels;

  return *this;
}

inline bool ALEScreen::equals(const ALEScreen &rhs) const {
  return (m_rows == rhs.m_rows &&
          m_columns == rhs.m_columns &&
          (memcmp(&m_pixels[0], &rhs.m_pixels[0], arraySize()) == 0) );
}

// pixel accessors, (row, column)-ordered
inline pixel_t ALEScreen::get(int r, int c) const {
  // Perform some bounds-checking
  assert (r >= 0 && r < m_rows && c >= 0 && c < m_columns);
  return m_pixels[r * m_columns + c];
}

inline pixel_t* ALEScreen::pixel(int r, int c) {
  // Perform some bounds-checking
  assert (r >= 0 && r < m_rows && c >= 0 && c < m_columns);
  return &m_pixels[r * m_columns + c];
}

// Access a whole row
inline pixel_t* ALEScreen::getRow(int r) const {
  assert (r >= 0 && r < m_rows);
  return const_cast<pixel_t*>(&m_pixels[r * m_columns]);
}


#endif // __ALE_SCREEN_HPP__

