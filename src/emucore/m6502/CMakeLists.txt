add_subdirectory(src/bspf)

target_include_directories(ale PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)
target_sources(ale
  PRIVATE
    src/Device.cxx
    src/M6502.cxx
    src/M6502Hi.cxx
    src/M6502Low.cxx
    src/NullDev.cxx
    src/System.cxx
)
