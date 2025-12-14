# WASM interface using Embind

# WASM executable target
# Include ale_interface.cpp since it's not part of the ale OBJECT library
add_executable(ale-wasm
    ale_wasm_interface.cpp
    ${PROJECT_SOURCE_DIR}/src/ale/ale_interface.cpp
)

# Link against the core ALE object library
target_link_libraries(ale-wasm PRIVATE ale)

# Add Emscripten system include directory for embind headers
target_include_directories(ale-wasm PRIVATE
    $ENV{EMSDK}/upstream/emscripten/system/include
)

# Emscripten-specific linker flags
set(EMSCRIPTEN_LINK_FLAGS "")
list(APPEND EMSCRIPTEN_LINK_FLAGS
    "-lembind"
    "--bind"
    "-sWASM=1"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME='createALEModule'"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sINITIAL_MEMORY=33554432"        # 32MB initial
    "-sMAXIMUM_MEMORY=134217728"       # 128MB max
    "-sEXPORTED_FUNCTIONS=['_malloc','_free']"
    "-sEXPORTED_RUNTIME_METHODS=['FS','cwrap','ccall']"
    "-sUSE_SDL=2"
    "-sDISABLE_EXCEPTION_CATCHING=0"
    "--pre-js ${CMAKE_CURRENT_SOURCE_DIR}/pre.js"
    "--post-js ${CMAKE_CURRENT_SOURCE_DIR}/post.js"
)

# Option to preload ROMs into the WASM virtual filesystem
# This creates an ale.data file with all ROMs embedded (~640KB)
option(WASM_PRELOAD_ROMS "Preload ROM files into WASM build" ON)
if(WASM_PRELOAD_ROMS)
    set(ROMS_DIR "${PROJECT_SOURCE_DIR}/src/ale/python/roms")
    if(EXISTS "${ROMS_DIR}")
        list(APPEND EMSCRIPTEN_LINK_FLAGS
            "--preload-file ${ROMS_DIR}@/roms"
        )
        message(STATUS "WASM: Preloading ROMs from ${ROMS_DIR}")
    else()
        message(WARNING "WASM: ROMs directory not found at ${ROMS_DIR}. "
                        "Run scripts/download_unpack_roms.sh to download ROMs. "
                        "Building without preloaded ROMs - you'll need to use loadROMFromURL().")
    endif()
endif()

# Add optimization flags for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND EMSCRIPTEN_LINK_FLAGS
        "-O3"
        "-sASSERTIONS=0"
        "-flto"
    )
else()
    list(APPEND EMSCRIPTEN_LINK_FLAGS
        "-O0"
        "-sASSERTIONS=2"
        "-sSTACK_OVERFLOW_CHECK=2"
        "-g4"
        "-gsource-map"
    )
endif()

# Convert list to string
string(REPLACE ";" " " EMSCRIPTEN_LINK_FLAGS_STR "${EMSCRIPTEN_LINK_FLAGS}")

# Set target properties
set_target_properties(ale-wasm PROPERTIES
    OUTPUT_NAME "ale"
    SUFFIX ".js"
    LINK_FLAGS "${EMSCRIPTEN_LINK_FLAGS_STR}"
)

# Install WASM artifacts
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ale.js
    ${CMAKE_CURRENT_BINARY_DIR}/ale.wasm
    DESTINATION wasm
)

# Install ROM data file if preloaded
if(WASM_PRELOAD_ROMS)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ale.data
        DESTINATION wasm
        OPTIONAL
    )
endif()

# Also install source maps if building in debug mode
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ale.wasm.map
        DESTINATION wasm
        OPTIONAL
    )
endif()
