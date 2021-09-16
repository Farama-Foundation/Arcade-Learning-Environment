if (TARGET ale-py)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  add_test(NAME ale-py COMMAND Python3::Interpreter -m pytest ${CMAKE_SOURCE_DIR})
  set_tests_properties(ale-py PROPERTIES
    ENV PYTHONPATH=${CMAKE_BINARY_DIR}/src
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src)
endif()
