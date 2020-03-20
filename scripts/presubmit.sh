#!/bin/bash

# Trailing whitespace in a line.
find -type f -name "*.hpp" -o -name "*.cpp" | xargs grep -ne '\s$' | sed 's/$/<--- trailing whitespace/' | grep . && exit 1
[[ $? -eq 0 ]] && exit 1

# Missing newline
find -type f -name "*.hpp" -o -name "*.cpp" | while read f; do [[ -z $(tail -c 1 ${f}) ]] || { echo "File ${f} is missing a final newline"; exit 1; } done
[[ $? -eq 0 ]] || exit 1

# Trailing empty lines
find -type f -name "*.hpp" -o -name "*.cpp" | while read f; do [[ $(tail -c 2 ${f} | wc -l) -le 1 ]] || { echo "File ${f} has trailing empty lines"; exit 1; } done
[[ $? -eq 0 ]] || exit 1

exit 0
