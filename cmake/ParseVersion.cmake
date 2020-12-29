# Version parsing.
#
# Parse PREFIX_DEFAULT_VERSION (MAJOR.MINOR.PATCH) from version.txt
# Set PREFIX_VERSION_GIT_SHA to the current GIT short SHA.
# Parse PREFIX_VERSION_{MAJOR, MINOR, PATCH} from PREFIX_DEFAULT_VERSION
# Parse ENV{PREFIX_BUILD_VERSION} if we are forcing the version e.g., in CD.
#
# PROJECT_VERSION is set to PREFIX_DEFAULT_VERSION as CMake is limited
# in what PROJECT_VERSION can represent. It can't handle full semver.
# TODO: Revisit this in future versions of CMake
#
function(parse_version filename)
  # Parse PREFIX arg
  cmake_parse_arguments(PARSE_VERSION "" "PREFIX" "" ${ARGN})

  # Read version from filename + strip whitespace
  file(READ ${filename} _DEFAULT_VERSION)
  string(STRIP "${_DEFAULT_VERSION}" _DEFAULT_VERSION)

  # Parse MAJOR.MINOR.PATCH
  string(REGEX MATCH "^([0-9]+)\\.?([0-9]+)\\.?([0-9]+)$" V ${_DEFAULT_VERSION})
  if(CMAKE_MATCH_COUNT LESS 3)
    message(
      FATAL_ERROR "Version number ${_DEFAULT_VERSION} has an invalid format")
  endif()

  set(_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(_VERSION_MINOR ${CMAKE_MATCH_2})
  set(_VERSION_PATCH ${CMAKE_MATCH_3})

  # Allow specifying custom version,
  # e.g., 1.0.0-alpha0, 1.0.0-nightly20201012
  #
  # with post patches for Python, see setup.py
  if(DEFINED ENV{${PARSE_VERSION_PREFIX}_BUILD_VERSION})
    set(_VERSION "$ENV{${PREFIX}_BUILD_VERSION}")
  else()
    set(_VERSION "${_DEFAULT_VERSION}")
  endif()

  # Find SHA
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE _VERSION_GIT_SHA
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(NOT _VERSION_GIT_SHA)
    set(_VERSION_GIT_SHA "unknown")
  endif()

  set(${PARSE_VERSION_PREFIX}_DEFAULT_VERSION
      ${_DEFAULT_VERSION}
      PARENT_SCOPE)
  set(${PARSE_VERSION_PREFIX}_VERSION
      ${_VERSION}
      PARENT_SCOPE)

  set(${PARSE_VERSION_PREFIX}_VERSION_MAJOR
      ${_VERSION_MAJOR}
      PARENT_SCOPE)
  set(${PARSE_VERSION_PREFIX}_VERSION_MINOR
      ${_VERSION_MINOR}
      PARENT_SCOPE)
  set(${PARSE_VERSION_PREFIX}_VERSION_PATCH
      ${_VERSION_PATCH}
      PARENT_SCOPE)

  set(${PARSE_VERSION_PREFIX}_VERSION_GIT_SHA
      ${_VERSION_GIT_SHA}
      PARENT_SCOPE)
endfunction()
