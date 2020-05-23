target_include_directories(ale PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)
target_sources(ale
  PRIVATE
    src/bspf.hxx
)
