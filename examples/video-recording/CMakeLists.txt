cmake_minimum_required(VERSION 3.14)

project(video-recording)

find_package(ale REQUIRED)

add_executable(videoRecordingExample videoRecordingExample.cpp)
target_link_libraries(videoRecordingExample ale::ale-lib)
