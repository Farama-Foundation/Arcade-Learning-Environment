#!/bin/bash

# define some directories
base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)/.."
unpack_dir="${base_dir}/unpack_dir"
target_dir="${base_dir}/src/ale/python/roms"

file_url="https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64"
expected_checksum="02ca777c16476a72fa36680a2ba78f24c3ac31b2155033549a5f37a0653117de"
temp_file="Roms.tar.gz.b64"

# make the directory where we will do the unpacking
mkdir $unpack_dir

# Download the ROMs
curl -o "$temp_file" "$file_url"

# Compute the SHA256 checksum of the ROMs file and compare with the expected checksum
if [[ "$(uname)" == "Darwin" ]]; then
    computed_checksum=$(shasum -a 256 "$temp_file" | awk '{ print $1 }')
else
    computed_checksum=$(sha256sum "$temp_file" | awk '{ print $1 }')
fi

if [ "$computed_checksum" == "$expected_checksum" ]; then
    # Decode the base64 file and extract the tar.gz content
    cat "$temp_file" \
      | base64 --decode \
      | tar -xzf - -C "$unpack_dir"

    # Clean up the temporary file
    rm -f "$temp_file"
else
    echo "Checksum verification failed! Exiting."
    exit 1  # Stop the script on checksum failure
fi

# move the ROMs out into a roms folder
mv ${unpack_dir}/ROM/*/*.bin $target_dir

# clean up
rm -r $unpack_dir
