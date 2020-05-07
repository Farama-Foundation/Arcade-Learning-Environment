add_subdirectory(supported)

target_sources(ale PRIVATE
        Roms.cpp
        RomSettings.cpp
        RomSettings2P.cpp
        RomSettings4P.cpp
        RomUtils.cpp
)
