#!/bin/bash

this_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
curl \
  "https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64" \
  -o "${this_directory}/src/python/roms/Roms.tar.gz.b64"
