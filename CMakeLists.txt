cmake_minimum_required (VERSION 2.6)
project(ale)
set(ALEVERSION "0.6")


option(USE_SDL "Use SDL" OFF)
option(USE_RLGLUE "Use RL-Glue" OFF)
option(BUILD_EXAMPLES "Build Example Agents" ON)
option(BUILD_CPP_LIB "Build C++ Shared Library" ON)
option(BUILD_CLI "Build ALE Command Line Interface" ON)
option(BUILD_C_LIB "Build ALE C Library (needed for Python interface)" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wunused -fPIC -O3 -fomit-frame-pointer -D__STDC_CONSTANT_MACROS")
add_definitions(-DHAVE_INTTYPES)
set(LINK_LIBS z)

if(USE_RLGLUE)
  add_definitions(-D__USE_RLGLUE)
  list(APPEND LINK_LIBS rlutils rlgluenetdev)
endif()

if(USE_SDL)
  add_definitions(-D__USE_SDL)
  add_definitions(-DSOUND_SUPPORT)
  find_package(SDL)
  if(SDL_FOUND AND ${SDL_VERSION_STRING} VERSION_LESS 2)
    include_directories(${SDL_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${SDL_LIBRARY} ${SDL_MAIN_LIBRARY})
  else()
    MESSAGE("SDL 1.2 not found: You may need to manually edit CMakeLists.txt or run \"cmake -i\" to specify your SDL path.")
    # Uncomment below to specify the path to your SDL library and header file. Run "locate libSDL" and "locate SDL.h" if unsure.
    # link_directories(path_to_your_SDL)
    # include_directories(path_to_your_SDL_header)
    if(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Cocoa")
      list(APPEND LINK_LIBS sdl sdlmain)
    else()
      list(APPEND LINK_LIBS SDL)
    endif()
  endif()
endif()

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(MODULES common controllers emucore emucore/m6502/src emucore/m6502/src/bspf/src environment games games/supported external external/TinyMT)

foreach(module ${MODULES})
  file(GLOB module_sources ${SOURCE_DIR}/${module}/*.c)
  list(APPEND SOURCES ${module_sources})
  file(GLOB module_sources ${SOURCE_DIR}/${module}/*.c?[xp])
  list(APPEND SOURCES ${module_sources})
endforeach(module ${MODULES})

# OS-dependent specifics
if(APPLE)
  include_directories(/System/Library/Frameworks/vecLib.framework/Versions/Current/Headers)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

if(WINDOWS OR MINGW)
  list(APPEND SOURCES ${SOURCE_DIR}/os_dependent/SettingsWin32.cxx ${SOURCE_DIR}/os_dependent/OSystemWin32.cxx ${SOURCE_DIR}/os_dependent/FSNodeWin32.cxx)
else()
  list(APPEND SOURCES ${SOURCE_DIR}/os_dependent/SettingsUNIX.cxx ${SOURCE_DIR}/os_dependent/OSystemUNIX.cxx ${SOURCE_DIR}/os_dependent/FSNodePOSIX.cxx)
  SET(BIN_INSTALL_DIR "bin")
  SET(LIB_SUFFIX "" CACHE STRING "Define suffix of directory name (32/64)" )
  SET(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}" CACHE STRING "Library directory name")
  SET(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include" CACHE STRING "Headers directory name")
  SET(PKGCONFIG_INSTALL_DIR "${LIB_INSTALL_DIR}/pkgconfig/" CACHE STRING "Base directory for pkgconfig files")

endif()

# List and set the install targets for the headers, generate and install the pkgconfig file
if(UNIX)

  INSTALL(FILES ${SOURCE_DIR}/os_dependent/SettingsUNIX.hxx ${SOURCE_DIR}/os_dependent/SettingsWin32.hxx ${SOURCE_DIR}/os_dependent/OSystemUNIX.hxx ${SOURCE_DIR}/os_dependent/OSystemWin32.hxx DESTINATION ${INCLUDE_INSTALL_DIR}/${PROJECT_NAME}/os_dependent)
  file(GLOB module_headers ${SOURCE_DIR}/*.h?[xp])
  foreach(header ${module_headers})
         INSTALL(FILES ${header} DESTINATION ${INCLUDE_INSTALL_DIR}/${PROJECT_NAME})
  endforeach(header ${HEADERS})

  foreach(module ${MODULES})
    file(GLOB module_headers ${SOURCE_DIR}/${module}/*.h)
    foreach(header ${module_headers})
         INSTALL(FILES ${header} DESTINATION ${INCLUDE_INSTALL_DIR}/${PROJECT_NAME}/${module}/)
    endforeach(header ${HEADERS})

    file(GLOB module_headers ${SOURCE_DIR}/${module}/*.h?[xp])
    foreach(header ${module_headers})
         INSTALL(FILES ${header} DESTINATION ${INCLUDE_INSTALL_DIR}/${PROJECT_NAME}/${module}/)
    endforeach(header ${HEADERS})
  endforeach(module ${MODULES})

  ###################################
  # Pkg-config stuff
  ###################################

  FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc
"
Name: ${PROJECT_NAME}
Description: The Arcade Learning Environment (ALE) - a platform for AI research.
URL: http://www.arcadelearningenvironment.org/
Version: ${ALEVERSION}
Requires:
Libs: -L${LIB_INSTALL_DIR} -lale
Cflags: -I${INCLUDE_INSTALL_DIR}
"
)


INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc
      DESTINATION ${PKGCONFIG_INSTALL_DIR})

endif(UNIX)

include_directories(
  ${SOURCE_DIR}
  ${SOURCE_DIR}/common
  ${SOURCE_DIR}/controllers
  ${SOURCE_DIR}/emucore
  ${SOURCE_DIR}/emucore/m6502/src
  ${SOURCE_DIR}/emucore/m6502/src/bspf/src
  ${SOURCE_DIR}/environment
  ${SOURCE_DIR}/games
  ${SOURCE_DIR}/games/supported
  ${SOURCE_DIR}/os_dependent
  ${SOURCE_DIR}/external
  ${SOURCE_DIR}/external/TinyMT
)

if(NOT BUILD_CPP_LIB AND BUILD_EXAMPLES)
  set(BUILD_CPP_LIB ON)
  MESSAGE("Enabling C++ library to support examples.")
endif()

if(BUILD_CPP_LIB)
  add_library(ale-lib SHARED ${SOURCE_DIR}/ale_interface.cpp ${SOURCES})
  set_target_properties(ale-lib PROPERTIES OUTPUT_NAME ale)
  set_target_properties(ale-lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  if(UNIX)
    install(TARGETS ale-lib
      DESTINATION ${LIB_INSTALL_DIR})
  endif()
  target_link_libraries(ale-lib ${LINK_LIBS})
endif()

if(BUILD_CLI)
  add_executable(ale-bin ${SOURCE_DIR}/main.cpp ${SOURCE_DIR}/ale_interface.cpp ${SOURCES})
  set_target_properties(ale-bin PROPERTIES OUTPUT_NAME ale)
  set_target_properties(ale-bin PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  if(UNIX)
    install(TARGETS ale-bin
      DESTINATION ${BIN_INSTALL_DIR})
  endif()
  target_link_libraries(ale-bin ${LINK_LIBS})
endif()

if(BUILD_C_LIB)
  add_library(ale-c-lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ale_python_interface/ale_c_wrapper.cpp ${SOURCE_DIR}/ale_interface.cpp ${SOURCES})
  set_target_properties(ale-c-lib PROPERTIES OUTPUT_NAME ale_c)
  set_target_properties(ale-c-lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ale_python_interface)
  if(UNIX)
    install(TARGETS ale-c-lib
      DESTINATION ${LIB_INSTALL_DIR})
  endif()
  target_link_libraries(ale-c-lib ${LINK_LIBS})
endif()

if(BUILD_EXAMPLES)
  # Shared library example.
  link_directories(${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(sharedLibraryInterfaceExample ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/sharedLibraryInterfaceExample.cpp)
  set_target_properties(sharedLibraryInterfaceExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  set_target_properties(sharedLibraryInterfaceExample PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-sharedLibraryInterfaceExample)

  target_link_libraries(sharedLibraryInterfaceExample ale)
  target_link_libraries(sharedLibraryInterfaceExample ${LINK_LIBS})
  add_dependencies(sharedLibraryInterfaceExample ale-lib)

  # Fifo interface example.
  link_directories(${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(fifoInterfaceExample ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/fifoInterfaceExample.cpp)
  set_target_properties(fifoInterfaceExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  set_target_properties(fifoInterfaceExample PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-fifoInterfaceExample)
  target_link_libraries(fifoInterfaceExample ale)
  target_link_libraries(fifoInterfaceExample ${LINK_LIBS})
  add_dependencies(fifoInterfaceExample ale-lib)

  add_executable(sharedLibraryInterfaceWithModesExample ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/sharedLibraryInterfaceWithModesExample.cpp)
  set_target_properties(sharedLibraryInterfaceWithModesExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  set_target_properties(sharedLibraryInterfaceWithModesExample PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-sharedLibraryInterfaceWithModesExample)
  target_link_libraries(sharedLibraryInterfaceWithModesExample ale)
  target_link_libraries(sharedLibraryInterfaceWithModesExample ${LINK_LIBS})
  add_dependencies(sharedLibraryInterfaceWithModesExample ale-lib)

  # Example showing how to record an Atari 2600 video.
  if (USE_SDL)
    add_executable(videoRecordingExample ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/videoRecordingExample.cpp)
    set_target_properties(videoRecordingExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
    set_target_properties(videoRecordingExample PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-videoRecordingExample)
    target_link_libraries(videoRecordingExample ale)
    target_link_libraries(videoRecordingExample ${LINK_LIBS})
    add_dependencies(videoRecordingExample ale-lib)
  endif()
endif()

if(USE_RLGLUE)
  add_executable(RLGlueAgent ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/RLGlueAgent.c)
  set_target_properties(RLGlueAgent PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  set_target_properties(RLGlueAgent PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-RLGlueAgent)
  target_link_libraries(RLGlueAgent rlutils)
  target_link_libraries(RLGlueAgent rlagent)
  target_link_libraries(RLGlueAgent rlgluenetdev)

  add_executable(RLGlueExperiment ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/RLGlueExperiment.c)
  set_target_properties(RLGlueExperiment PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  set_target_properties(RLGlueExperiment PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-RLGlueExperiment)
  target_link_libraries(RLGlueExperiment rlutils)
  target_link_libraries(RLGlueExperiment rlexperiment)
  target_link_libraries(RLGlueExperiment rlgluenetdev)

endif()


########### Add uninstall target ###############
  FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
"
IF(NOT EXISTS \"@CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt\")
  MESSAGE(FATAL_ERROR \"Cannot find install manifest: @CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt\")
ENDIF(NOT EXISTS \"@CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt\")

FILE(READ \"@CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt\" files)
STRING(REGEX REPLACE \"\\n\" \";\" files \"\${files}\")
FOREACH(file \${files})
  MESSAGE(STATUS \"Uninstalling \"\$ENV{DESTDIR}\${file}\"\")
  IF(EXISTS \"\$ENV{DESTDIR}\${file}\")
    EXEC_PROGRAM(
      \"@CMAKE_COMMAND@\" ARGS \"-E remove \"\$ENV{DESTDIR}\${file}\"\"
      OUTPUT_VARIABLE rm_out
      RETURN_VALUE rm_retval
      )
    IF(NOT \"\${rm_retval}\" STREQUAL 0)
      MESSAGE(FATAL_ERROR \"Problem when removing \"\$ENV{DESTDIR}\${file}\"\")
    ENDIF(NOT \"\${rm_retval}\" STREQUAL 0)
  ELSE(EXISTS \"\$ENV{DESTDIR}\${file}\")
    MESSAGE(STATUS \"File \"\$ENV{DESTDIR}\${file}\" does not exist.\")
  ENDIF(EXISTS \"\$ENV{DESTDIR}\${file}\")
ENDFOREACH(file)
")

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  COMMAND rm -rf ${INCLUDE_INSTALL_DIR}/ale)


