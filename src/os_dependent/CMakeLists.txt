if (WIN32)
  target_sources(ale
    PRIVATE
      FSNodeWin32.cxx
      OSystemWin32.cxx
      SettingsWin32.cxx
  )
elseif (UNIX)
  target_sources(ale
    PRIVATE
      FSNodePOSIX.cxx
      OSystemUNIX.cxx
      SettingsUNIX.cxx
  )
endif()
