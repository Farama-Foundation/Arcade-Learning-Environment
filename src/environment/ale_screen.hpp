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
    pixel_t *getArray() const { return (pixel_t*)m_pixels.get(); }

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

    std::auto_ptr<pixel_t> m_pixels; 
};

inline ALEScreen::ALEScreen(int h, int w):
  m_rows(h),
  m_columns(w),
  // Create a pixel array of the requisite size
  m_pixels(new pixel_t[m_rows * m_columns]) {
}

inline ALEScreen::ALEScreen(const ALEScreen &rhs):
  m_rows(rhs.m_rows),
  m_columns(rhs.m_columns),
  m_pixels(new pixel_t[m_rows * m_columns]) {

  // Copy data over
  memcpy(m_pixels.get(), rhs.m_pixels.get(), arraySize());
}

inline ALEScreen& ALEScreen::operator=(const ALEScreen &rhs) {
  // If array dimensions are mistmatched, must reallocate the array
  if (m_rows != rhs.m_rows || m_columns != rhs.m_columns) {
    m_rows = rhs.m_rows;
    m_columns = rhs.m_columns;

    m_pixels.reset(new pixel_t[m_rows * m_columns]);
  }
  
  // Copy data over 
  memcpy(m_pixels.get(), rhs.m_pixels.get(), arraySize());

  return *this;
}

inline bool ALEScreen::equals(const ALEScreen &rhs) const {
  return (m_rows == rhs.m_rows &&
          m_columns == rhs.m_columns &&
          (memcmp(m_pixels.get(), rhs.m_pixels.get(), arraySize()) == 0) );
}

// pixel accessors, (row, column)-ordered
inline pixel_t ALEScreen::get(int r, int c) const {
  // Perform some bounds-checking
  assert (r >= 0 && r < m_rows && c >= 0 && c < m_columns);
  return m_pixels.get()[r * m_columns + c];
}

inline pixel_t* ALEScreen::pixel(int r, int c) {
  // Perform some bounds-checking
  assert (r >= 0 && r < m_rows && c >= 0 && c < m_columns);
  return &(m_pixels.get()[r * m_columns + c]);
}

// Access a whole row
inline pixel_t* ALEScreen::getRow(int r) const {
  assert (r >= 0 && r < m_rows);
  return &(m_pixels.get()[r * m_columns]);
}


#endif // __ALE_SCREEN_HPP__

