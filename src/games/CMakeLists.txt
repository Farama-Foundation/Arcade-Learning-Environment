add_subdirectory(supported)
target_sources(ale
  PRIVATE
    Roms.cpp
    RomSettings.cpp
    RomUtils.cpp
)
