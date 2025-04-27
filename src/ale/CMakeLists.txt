# Include src/ and cmake binary directory (for version.hpp)
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})

# Project specific target properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Compiler options
add_compile_options(
  "$<$<CONFIG:RELEASE>:-O3>"
  "$<$<CONFIG:DEBUG>:-O0>"
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>"
  "$<$<CXX_COMPILER_ID:MSVC>:/W4>")

# ale object containing source files
add_library(ale OBJECT)

# Dependencies
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(ale
  PRIVATE
    ZLIB::ZLIB
    Threads::Threads)

if(SDL_SUPPORT)
  option(SDL_DYNLOAD "Dynamically load SDL" OFF)
  find_package(SDL2 CONFIG REQUIRED)

  # VCPKG doesn't include SDL2_INCLUDE_DIRS
  if(VCPKG_TOOLCHAIN)
    find_path(SDL2_INCLUDE_DIRS SDL.h PATH_SUFFIXES SDL2)
  endif()

  target_compile_definitions(ale
    PUBLIC
      ALE_SDL_SUPPORT
    PRIVATE
      SDL_SUPPORT
      $<$<BOOL:${SDL_DYNLOAD}>:SDL_DYNLOAD>)
  target_link_libraries(ale PRIVATE $<$<NOT:$<BOOL:${SDL_DYNLOAD}>>:SDL2::SDL2>)
  target_include_directories(ale PRIVATE ${SDL2_INCLUDE_DIRS})
endif()

configure_file ("version.hpp.in" "version.hpp")

# Add submodules
add_subdirectory(common)
add_subdirectory(emucore)
add_subdirectory(environment)
add_subdirectory(games)

# C++ Library
if (BUILD_CPP_LIB OR BUILD_PYTHON_LIB)
  add_library(ale-lib ale_interface.cpp)
  set_target_properties(ale-lib PROPERTIES OUTPUT_NAME ale)
  target_link_libraries(ale-lib PUBLIC ale)

  if (BUILD_VECTOR_LIB)
    find_package(OpenCV CONFIG REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    target_include_directories(ale PUBLIC external)

    add_subdirectory(vector)

    target_link_libraries(ale PRIVATE ${OpenCV_LIBS})
  endif()
endif()

# Python Library
if (BUILD_PYTHON_LIB)
  add_subdirectory(python)
endif()

# Install cpp-lib and CMake config & pkg-config
if (UNIX AND BUILD_CPP_LIB)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(TARGETS ale ale-lib
          EXPORT ale-export
          INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Install Header files
  install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
          FILES_MATCHING
          REGEX "\.h((pp)?|(xx?))$"
          REGEX "os_dependent\/.*(Win32)\.(h|h(xx)?|h(pp)?|c(xx)?|c(pp)?)$" EXCLUDE)

  # Install configured version file
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.hpp
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

  # Export cmake targets
  install(EXPORT ale-export
          FILE ${PROJECT_NAME}-targets.cmake
          NAMESPACE ${PROJECT_NAME}::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  # Write config-version
  write_basic_package_version_file(
    ${PROJECT_NAME}-config-version.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion)

  # Configure installable cmake config
  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
    ${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  # Install config-version and config
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
endif()
