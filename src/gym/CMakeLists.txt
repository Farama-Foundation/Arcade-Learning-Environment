find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Copy over the Python source files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILES_MATCHING
        PATTERN "**/*.py")
