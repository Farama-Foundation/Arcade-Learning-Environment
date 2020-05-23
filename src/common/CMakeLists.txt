target_include_directories(ale PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_sources(ale PRIVATE
        ColourPalette.cpp
        Constants.cpp
        display_screen.cpp
        Log.cpp
        Palettes.hpp
        ScreenExporter.cpp
        SoundExporter.cpp
        SoundNull.cxx
        SoundSDL.cxx
        Version.hxx
        VideoModeList.hxx
        Array.hxx
)
