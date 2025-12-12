#!/bin/bash
# Build ALE WASM and package for NPM and standalone distribution
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$BASE_DIR/build-wasm"
PACKAGE_DIR="$BASE_DIR/packages/wasm"
# Find Emscripten SDK (EMSDK_PATH takes priority)
if [ -n "$EMSDK_PATH" ]; then
    : # Use provided EMSDK_PATH
elif [ -n "$EMSDK" ] && [ -f "$EMSDK/emsdk_env.sh" ]; then
    EMSDK_PATH="$EMSDK"
elif [ -f "$HOME/emsdk/emsdk_env.sh" ]; then
    EMSDK_PATH="$HOME/emsdk"
elif [ -f "/opt/emsdk/emsdk_env.sh" ]; then
    EMSDK_PATH="/opt/emsdk"
else
    echo "Error: Emscripten SDK not found. Set EMSDK_PATH or install to ~/emsdk"
    exit 1
fi

source "$EMSDK_PATH/emsdk_env.sh"
VERSION=$(cat "$BASE_DIR/version.txt" | tr -d '[:space:]')

# Configure and build
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

emcmake cmake "$BASE_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WASM_LIB=ON \
    -DWASM_PRELOAD_ROMS=ON \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_TARGET_TRIPLET=wasm32-emscripten \
    -DVCPKG_OVERLAY_TRIPLETS="$BASE_DIR/cmake/custom-triplets" \
    -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE="$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"

emmake make ale-wasm -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

WASM_DIR="$BUILD_DIR/src/ale/wasm"

# Create dist directories
mkdir -p "$BASE_DIR/dist/npm" "$BASE_DIR/dist/ale-wasm-$VERSION"

# NPM package: copy package files + built WASM
cp "$PACKAGE_DIR/package.json" "$PACKAGE_DIR/ale.d.ts" "$PACKAGE_DIR/README.md" "$BASE_DIR/dist/npm/"
cp "$WASM_DIR/ale.js" "$WASM_DIR/ale.wasm" "$WASM_DIR/ale.data" "$BASE_DIR/dist/npm/"
cp "$BASE_DIR/LICENSE.md" "$BASE_DIR/dist/npm/LICENSE"

# Update version in package.json
sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$BASE_DIR/dist/npm/package.json"
rm -f "$BASE_DIR/dist/npm/package.json.bak"

# Standalone package
STANDALONE_DIR="$BASE_DIR/dist/ale-wasm-$VERSION"
cp "$WASM_DIR/ale.js" "$WASM_DIR/ale.wasm" "$WASM_DIR/ale.data" "$STANDALONE_DIR/"
cp "$PACKAGE_DIR/ale.d.ts" "$PACKAGE_DIR/README.md" "$STANDALONE_DIR/"

cd "$BASE_DIR/dist"
zip -r "ale-wasm-$VERSION.zip" "ale-wasm-$VERSION"

echo ""
echo "Done! Packages created:"
echo "  NPM:        $BASE_DIR/dist/npm/"
echo "  Standalone: $BASE_DIR/dist/ale-wasm-$VERSION.zip"

