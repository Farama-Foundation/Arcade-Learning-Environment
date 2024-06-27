find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.12.0)
FetchContent_MakeAvailable(pybind11)

add_library(ale-py MODULE ale_python_interface.cpp)
# Depend on the ALE and pybind11 module
target_link_libraries(ale-py PUBLIC ale ale-lib)
target_link_libraries(ale-py PRIVATE pybind11::module)

# The native module is imported as _ale_py
# Make sure to set the proper prefix and extension from pybind11
set_target_properties(ale-py PROPERTIES
    OUTPUT_NAME _ale_py
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")

# To test ale-py we need to make sure that all generated resources
# are in CMAKE_CURRENT_BINARY_DIR, i.e., build/src/python.
# MSVC adds the config to the path, i.e., Debug/Release.
# There's a hack in CMake where if a generator expression is used
# this Debug/Release suffix won't be added. We can append a null generator
# expression to the path to force all outputs CMAKE_CURRENT_BINARY_DIR.
if (MSVC)
  set_target_properties(ale-py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}$<0:>
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}$<0:>
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}$<0:>)
endif()

# Copy over the Python source files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILES_MATCHING
        PATTERN "**/*.py")
# We symlink build/src/python -> build/ale_py
# So we can have proper module discovery for testing
# See tests/CMakeLists.txt
add_custom_command(TARGET ale-py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/src/ale_py)


# If we're dynamically loading SDL with Python we'll be building a wheel
# so we should prepare SDL for distribution. Add rpath and copy over
# the dynamic library. auditwheel will take care of ensuring
# corss-platform compatability on macOS and Linux.
if (SDL_SUPPORT AND SDL_DYNLOAD)
    set_target_properties(ale-py PROPERTIES
        INSTALL_RPATH_USE_ORIGIN TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        MACOSX_RPATH TRUE
        INSTALL_RPATH
        "$<$<PLATFORM_ID:Darwin>:@loader_path>$<$<PLATFORM_ID:Linux>:\$ORIGIN>")

    # Define our SDL2 distribution library name for dynamic loading
    target_compile_definitions(ale
        PRIVATE SDL2_LIBRARY_NAME="$<TARGET_FILE_NAME:SDL2::SDL2>")
    # Copy over SDL2 dist. library
    add_custom_command(TARGET ale-py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL2::SDL2>
            $<TARGET_FILE_DIR:ale-py>)
endif()
