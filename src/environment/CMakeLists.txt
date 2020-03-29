target_include_directories(ale PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_sources(ale
  PRIVATE
    ale_state.cpp
    phosphor_blend.cpp
    stella_environment.cpp
    stella_environment_wrapper.cpp
)
