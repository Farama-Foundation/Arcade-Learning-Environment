find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_VER 2.13.1)
find_package(pybind11 ${PYBIND11_VER} QUIET)

if(NOT pybind11_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v${PYBIND11_VER})
    FetchContent_MakeAvailable(pybind11)
endif()

# If vector library is enabled, include the vector Python interface
# XLA Integration
if (BUILD_VECTOR_LIB AND BUILD_VECTOR_XLA_LIB)
#    message(STATUS "Python Executable: '${Python3_EXECUTABLE}'")
#    message(STATUS "Python3_SITELIB=${Python3_SITELIB}, Python3_SITEARCH=${Python3_SITEARCH}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-c"
        "import sys; sys.path.append(r'${Python3_SITELIB}'); sys.path.append(r'${Python3_SITEARCH}'); from jax import ffi; print(ffi.include_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE XLA_DIR
    )
#    message(STATUS "XLA include directory: '${XLA_DIR}'")

#    find_package(CUDAToolkit REQUIRED)

    add_library(ale-py MODULE ale_python_interface.cpp ale_vector_python_interface.cpp ale_vector_xla_interface.cpp)
    set_target_properties(ale-py PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
    target_include_directories(ale-py PUBLIC ${XLA_DIR})
    target_compile_definitions(ale-py PRIVATE BUILD_VECTOR_LIB BUILD_VECTOR_XLA_LIB)

elseif (BUILD_VECTOR_LIB)
    add_library(ale-py MODULE ale_python_interface.cpp ale_vector_python_interface.cpp)
    target_compile_definitions(ale-py PRIVATE BUILD_VECTOR_LIB)
else ()
    add_library(ale-py MODULE ale_python_interface.cpp)
endif()

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
