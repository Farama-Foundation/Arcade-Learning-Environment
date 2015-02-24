cmake_minimum_required (VERSION 2.6)
project(ale)

option(USE_SDL "Use SDL" OFF)
option(USE_RLGLUE "Use RL-Glue" OFF)
option(BUILD_EXAMPLES "Build Example Agents" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wunused -fPIC -O3 -fomit-frame-pointer")
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
  if(SDL_FOUND)
    include_directories(${SDL_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${SDL_LIBRARY} ${SDL_MAIN_LIBRARY})
  else()
    MESSAGE("SDL not found. You may need to manually specify your SDL path in CMakeLists.txt")
    # Specify the path to your SDL libraries. Run "locate libSDL" if unsure.
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
set(MODULES common controllers emucore emucore/m6502/src emucore/m6502/src/bspf/src environment games games/supported)

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

if(WINDOWS)
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
)

add_library(ale-lib SHARED ${SOURCE_DIR}/ale_interface.cpp ${SOURCES})
set_target_properties(ale-lib PROPERTIES OUTPUT_NAME ale)
add_executable(ale-bin ${SOURCE_DIR}/main.cpp ${SOURCE_DIR}/ale_interface.cpp ${SOURCES})
set_target_properties(ale-bin PROPERTIES OUTPUT_NAME ale)

target_link_libraries(ale-lib ${LINK_LIBS})
target_link_libraries(ale-bin ${LINK_LIBS})

if(BUILD_EXAMPLES)
  link_directories(${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(sharedLibraryInterfaceExample ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples/sharedLibraryInterfaceExample.cpp)
  set_target_properties(sharedLibraryInterfaceExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/examples)
  target_link_libraries(sharedLibraryInterfaceExample ale)
  target_link_libraries(sharedLibraryInterfaceExample ${LINK_LIBS})
  add_dependencies(sharedLibraryInterfaceExample ale-lib)
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