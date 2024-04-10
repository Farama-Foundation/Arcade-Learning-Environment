#!/bin/bash

# define some directories
this_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
unpack_dir="${this_directory}/unpack_dir"
target_dir="${this_directory}/src/python/roms"

# make the directory where we will do the unpacking
mkdir $unpack_dir

curl "https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64" \
  | base64 --decode \
  | tar -xzf - -C $unpack_dir

# move the ROMs out into a roms folder
mv ${unpack_dir}/ROM/*/*.bin $target_dir

# clean up
rm $unpack_dir

echo "Extraction complete."
