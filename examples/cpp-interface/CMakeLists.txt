cmake_minimum_required(VERSION 3.14)

project(example-cpp-lib)

find_package(ale REQUIRED)

add_executable(sharedLibraryInterfaceExample sharedLibraryInterfaceExample.cpp)
target_link_libraries(sharedLibraryInterfaceExample ale::ale-lib)

add_executable(sharedLibraryInterfaceWithModesExample sharedLibraryInterfaceWithModesExample.cpp)
target_link_libraries(sharedLibraryInterfaceWithModesExample ale::ale-lib)
