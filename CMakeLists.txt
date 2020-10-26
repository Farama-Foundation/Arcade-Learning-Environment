cmake_minimum_required (VERSION 3.14)

project(ale VERSION 0.1.9
            DESCRIPTION "The Arcade Learning Environment (ALE) - a platform for AI research."
            HOMEPAGE_URL "http://www.arcadelearningenvironment.org"
            LANGUAGES CXX)

# Set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

add_subdirectory(src)
