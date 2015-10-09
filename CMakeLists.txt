cmake_minimum_required (VERSION 2.6)
project(ale)

option(USE_SDL "Use SDL" OFF)
option(USE_RLGLUE "Use RL-Glue" OFF)
option(BUILD_EXAMPLES "Build Example Agents" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wunused -fPIC -O3 -fomit-frame-pointer -D__STDC_CONSTANT_MACROS")
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
    # Uncomment below to specify the path to your SDL library. Run "locate libSDL" if unsure.
    # link_directories(path_to_your_SDL)
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
endif()

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

add_library(ale-lib SHARED ${SOURCE_DIR}/ale_interface.cpp ${SOURCES})
set_target_properties(ale-lib PROPERTIES OUTPUT_NAME ale)
set_target_properties(ale-lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(ale-bin ${SOURCE_DIR}/main.cpp ${SOURCE_DIR}/ale_interface.cpp ${SOURCES})
set_target_properties(ale-bin PROPERTIES OUTPUT_NAME ale)
set_target_properties(ale-bin PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_library(ale-c-lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ale_python_interface/ale_c_wrapper.cpp ${SOURCE_DIR}/ale_interface.cpp ${SOURCES})
set_target_properties(ale-c-lib PROPERTIES OUTPUT_NAME ale_c)
set_target_properties(ale-c-lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ale_python_interface)

target_link_libraries(ale-lib ${LINK_LIBS})
target_link_libraries(ale-bin ${LINK_LIBS})
target_link_libraries(ale-c-lib ${LINK_LIBS})

if(BUILD_EXAMPLES)
  link_directories(${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(sharedLibraryInterfaceExample ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/sharedLibraryInterfaceExample.cpp)
  set_target_properties(sharedLibraryInterfaceExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  target_link_libraries(sharedLibraryInterfaceExample ale)
  target_link_libraries(sharedLibraryInterfaceExample ${LINK_LIBS})
  add_dependencies(sharedLibraryInterfaceExample ale-lib)
  
  if (USE_SDL)
    add_executable(videoRecordingExample ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/videoRecordingExample.cpp)
    set_target_properties(videoRecordingExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
    target_link_libraries(videoRecordingExample ale)
    target_link_libraries(videoRecordingExample ${LINK_LIBS})
    add_dependencies(videoRecordingExample ale-lib)
  endif()
endif()

if(USE_RLGLUE)
  add_executable(RLGlueAgent ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/RLGlueAgent.c)
  set_target_properties(RLGlueAgent PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  target_link_libraries(RLGlueAgent rlutils)
  target_link_libraries(RLGlueAgent rlagent)
  target_link_libraries(RLGlueAgent rlgluenetdev)
  add_executable(RLGlueExperiment ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/RLGlueExperiment.c)
  set_target_properties(RLGlueExperiment PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  target_link_libraries(RLGlueExperiment rlutils)
  target_link_libraries(RLGlueExperiment rlexperiment)
  target_link_libraries(RLGlueExperiment rlgluenetdev)
endif()
