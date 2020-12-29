cmake_minimum_required (VERSION 3.14)

# Set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# VCPKG triplet and toolchain discovery
if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET
      "$ENV{VCPKG_DEFAULT_TRIPLET}"
      CACHE STRING "")
endif()

if(DEFINED ENV{VCPKG_INSTALLATION_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE
      "$ENV{VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

include(ParseVersion)
parse_version("version.txt" PREFIX ALE)

project(ale VERSION ${ALE_DEFAULT_VERSION}
            DESCRIPTION "The Arcade Learning Environment (ALE) - a platform for AI research."
            HOMEPAGE_URL "http://www.arcadelearningenvironment.org"
            LANGUAGES CXX)

# Main ALE src directory
add_subdirectory(src)

# Only include tests in the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  enable_testing()
  add_subdirectory(tests)
endif()
