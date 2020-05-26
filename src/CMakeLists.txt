option(USE_SDL "Use SDL" OFF)
option(BUILD_CPP_LIB "Build C++ Shared Library" ON)
option(BUILD_TEST_EXEC "Build C++ Shared Library" OFF)
option(BUILD_C_LIB "Build C Shared Library" OFF)


# Include src/ and cmake binary directory (for version.hpp)
include_directories(BEFORE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# ale object containing source files
add_library(ale OBJECT)

# Require C++11 for dependencies
target_compile_features(ale PUBLIC cxx_std_11)
target_compile_options(ale PUBLIC -O3 -fPIC)

# Compiler specific flags
target_compile_options(ale
  PUBLIC
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wunused>
)

# Dependencies
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(ale
  PRIVATE
    ZLIB::ZLIB
    Threads::Threads
)

if (USE_SDL)
  target_compile_definitions(ale
    PUBLIC
      -D__USE_SDL
      -DSOUND_SUPPORT
  )
  find_package(SDL REQUIRED)
  target_include_directories(ale PUBLIC ${SDL_INCLUDE_DIR})
  target_link_libraries(ale PUBLIC ${SDL_LIBRARY})
endif()

# Add submodules
add_subdirectory(common)
add_subdirectory(emucore)
add_subdirectory(os_dependent)
add_subdirectory(environment)
add_subdirectory(games)

find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_SHA
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()
if (NOT GIT_SHA)
  set(GIT_SHA "(unknown)")
endif()
configure_file ("version.hpp.in" "version.hpp")


if (BUILD_TEST_EXEC)
  add_executable(test_exec test_main.cpp ale_interface.cpp)
  set_target_properties(test_exec PROPERTIES OUTPUT_NAME test_exe)
  target_link_libraries(test_exec PUBLIC ale ${ZLIB_LIBRARIES})

endif()

# C++ Library
if (BUILD_CPP_LIB)
  add_library(ale-lib ale_interface.cpp)
  set_target_properties(ale-lib PROPERTIES OUTPUT_NAME ale)
  target_link_libraries(ale-lib PUBLIC ale)
endif()

if(BUILD_C_LIB)
  add_library(ale-c-lib SHARED ./../multi_agent_ale_py/ale_c_wrapper.cpp ale_interface.cpp)
  set_target_properties(ale-c-lib PROPERTIES OUTPUT_NAME ale_c)
  #set_target_properties(ale-c-lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/multi_agent_ale_py)
  if(PYTHON_MODULE_EXTENSION)
    set_target_properties(ale-c-lib PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")
  endif()
  if(UNIX)
    install(TARGETS ale-c-lib
      DESTINATION ${LIB_INSTALL_DIR})
  endif()
  target_link_libraries(ale-c-lib ale)
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
    COMPATIBILITY AnyNewerVersion
  )

  # Configure installable cmake config
  configure_package_config_file(
    ${CMAKE_MODULE_PATH}/${PROJECT_NAME}-config.cmake.in
    ${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  # Install config-version and config
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
endif()
