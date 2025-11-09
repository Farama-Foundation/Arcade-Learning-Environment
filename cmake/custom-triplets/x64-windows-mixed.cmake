set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)

# Note: VCPKG_CMAKE_SYSTEM_NAME should NOT be set for native Windows builds
# It's only needed for cross-compilation scenarios
# set(VCPKG_CMAKE_SYSTEM_NAME Windows)

# Use dynamic linking only for SDL, static for everything else
# This avoids DLL distribution issues while keeping SDL dynamic for wheel compatibility
if(PORT MATCHES "sdl")
  set(VCPKG_LIBRARY_LINKAGE dynamic)
else()
  set(VCPKG_LIBRARY_LINKAGE static)
endif()
