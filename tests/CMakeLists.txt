if (BUILD_PYTHON)
  add_custom_target(pytest
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:ale-py> ${PYTHON_EXECUTABLE} -m pytest --verbose
    DEPENDS ale-py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
  add_custom_target(tests DEPENDS pytest)
endif()
