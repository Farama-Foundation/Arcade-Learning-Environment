find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

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

# Use pybind11_add_module for proper scikit-build-core integration
pybind11_add_module(_ale_py MODULE ale_python_interface.cpp)
if (BUILD_VECTOR_LIB AND BUILD_VECTOR_XLA_LIB)
    # XLA Integration setup
    message(STATUS "Python Executable: '${Python_EXECUTABLE}'")
    message(STATUS "Python_SITELIB=${Python_SITELIB}, Python_SITEARCH=${Python_SITEARCH}")
    execute_process(
        COMMAND "${Python_EXECUTABLE}" "-c"
        "import sys; sys.path.append(r'${Python_SITELIB}'); sys.path.append(r'${Python_SITEARCH}'); from jax import ffi; print(ffi.include_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE XLA_DIR
    )
    message(STATUS "XLA include directory: '${XLA_DIR}'")

    target_sources(_ale_py PRIVATE ale_vector_python_interface.cpp ale_vector_xla_interface.cpp)
    target_include_directories(_ale_py PUBLIC ${XLA_DIR})
    target_compile_definitions(_ale_py PRIVATE BUILD_VECTOR_LIB BUILD_VECTOR_XLA_LIB)
    set_target_properties(_ale_py PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
elseif (BUILD_VECTOR_LIB)
    target_sources(_ale_py PRIVATE ale_vector_python_interface.cpp)
    target_compile_definitions(_ale_py PRIVATE BUILD_VECTOR_LIB)
endif()

target_link_libraries(_ale_py PUBLIC ale ale-lib)

# Install the compiled module
install(TARGETS _ale_py LIBRARY DESTINATION ale_py)

# Install Python source files
install(FILES
    __init__.py
    env.py
    registration.py
    vector_env.py
    py.typed
    DESTINATION ale_py)

# Install stub files if they exist
install(FILES
    __init__.pyi
    DESTINATION ale_py
    OPTIONAL)

# Install roms directory
install(DIRECTORY roms/
    DESTINATION ale_py/roms
    FILES_MATCHING PATTERN "*")

# If we're dynamically loading SDL with Python we'll be building a wheel
# so we should prepare SDL for distribution. Add rpath and copy over
# the dynamic library. auditwheel will take care of ensuring
# cross-platform compatibility on macOS and Linux.
if (SDL_SUPPORT AND SDL_DYNLOAD)
    set_target_properties(_ale_py PROPERTIES
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
    add_custom_command(TARGET _ale_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL2::SDL2>
            $<TARGET_FILE_DIR:_ale_py>)
    # Install SDL2 library alongside the module
    install(FILES $<TARGET_FILE:SDL2::SDL2> DESTINATION ale_py)
endif()
