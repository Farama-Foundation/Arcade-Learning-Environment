cmake_minimum_required (VERSION 3.14)

# Build the C++ shared library
option(BUILD_CPP_LIB "Build C++ Interface" ON)

# Build the native Python bindings using pybind11
option(BUILD_PYTHON_LIB "Build Python Interface" ON)

# Enable SDL for screen and audio support
option(SDL_SUPPORT "Enable SDL support" OFF)
# Append VCPKG manifest feature
if(SDL_SUPPORT)
  list(APPEND VCPKG_MANIFEST_FEATURES "sdl")
endif()

option(BUILD_VECTOR_LIB "Build Vector Interface" OFF)
if (BUILD_VECTOR_LIB)
  list(APPEND VCPKG_MANIFEST_FEATURES "vector")
  add_definitions(-DBUILD_VECTOR_LIB)
endif()

option(BUILD_VECTOR_XLA_LIB "Build with Vector XLA Support" OFF)
if (BUILD_VECTOR_XLA_LIB)
  add_definitions(-DBUILD_VECTOR_XLA_LIB)
endif()

# Sanitizer support for debugging (thread, address, undefined)
# Usage: -DENABLE_SANITIZER=thread or -DENABLE_SANITIZER=address
set(ENABLE_SANITIZER "" CACHE STRING "Enable sanitizer (thread, address, or empty to disable)")
set_property(CACHE ENABLE_SANITIZER PROPERTY STRINGS "" "thread" "address")

if(ENABLE_SANITIZER)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(WARNING "Sanitizers are best supported with Clang or GCC. Current compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  if(ENABLE_SANITIZER STREQUAL "thread")
    message(STATUS "Building with Thread Sanitizer (TSan)")
    add_compile_options(-fsanitize=thread -g -O1)
    add_link_options(-fsanitize=thread)
  elseif(ENABLE_SANITIZER STREQUAL "address")
    message(STATUS "Building with Address Sanitizer (ASan) + Undefined Behavior Sanitizer (UBSan)")
    add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=address -fsanitize=undefined)
  else()
    message(FATAL_ERROR "Invalid sanitizer '${ENABLE_SANITIZER}'. Choose 'thread' or 'address'.")
  endif()
endif()

# Set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Overlay VCPKG custom triplets
if(NOT DEFINED VCPKG_OVERLAY_TRIPLETS)
  set(VCPKG_OVERLAY_TRIPLETS
      "${CMAKE_MODULE_PATH}/custom-triplets"
      CACHE STRING "")
endif()

# Discover VCPKG default triplet
if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET
      "$ENV{VCPKG_DEFAULT_TRIPLET}"
      CACHE STRING "")
endif()

# Discover VCPKG toolchain
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  # VCPKG_ROOT is what Microsoft recommends,
  # VCPKG_INSTALLATION_ROOT is what's defined on Azure
  if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
  elseif(DEFINED ENV{VCPKG_INSTALLATION_ROOT})
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
  endif()
endif()

# Don't allow running cmake in root directory
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(FATAL_ERROR [=[
    Source and build directories cannot be the same.
    You should probably also remove CMakeFiles/ and CMakeCache.txt.
  ]=])
endif()

include(ParseVersion)
parse_version("version.txt" PREFIX ALE)

project(ale VERSION ${ALE_DEFAULT_VERSION}
            DESCRIPTION "The Arcade Learning Environment (ALE) - a platform for AI research."
            HOMEPAGE_URL "http://www.arcadelearningenvironment.org"
            LANGUAGES CXX)

# Main ALE src directory
add_subdirectory(src/ale)

# Only include tests in the main project
# if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#  enable_testing()
#  add_subdirectory(tests)
# endif()
