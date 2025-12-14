#!/bin/bash
# Helper script for running sanitizers locally
# Usage: ./scripts/run_sanitizers.sh [thread|address|valgrind-memcheck|valgrind-helgrind]

set -e

SANITIZER=${1:-thread}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running sanitizer: $SANITIZER"
echo "=========================================="

# Ensure ROMs are downloaded
if [ ! -d "src/ale/python/roms" ]; then
    echo "Downloading ROMs..."
    ./scripts/download_unpack_roms.sh
fi

case $SANITIZER in
    thread)
        echo "Building with Thread Sanitizer..."

        # On macOS, use Homebrew LLVM to avoid SIP restrictions
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                LLVM_PREFIX=$(brew --prefix llvm 2>/dev/null || echo "")
                if [ -n "$LLVM_PREFIX" ] && [ -d "$LLVM_PREFIX" ]; then
                    echo "Using Homebrew LLVM from $LLVM_PREFIX"
                    export CC="$LLVM_PREFIX/bin/clang"
                    export CXX="$LLVM_PREFIX/bin/clang++"

                    # Pass Homebrew LLVM library paths to CMake
                    # Use a single cmake.args with semicolon-separated values
                    pip install --verbose -e .[test] \
                        --config-settings=cmake.args="-DENABLE_SANITIZER=thread;-DCMAKE_EXE_LINKER_FLAGS=-L$LLVM_PREFIX/lib -Wl,-rpath,$LLVM_PREFIX/lib;-DCMAKE_SHARED_LINKER_FLAGS=-L$LLVM_PREFIX/lib -Wl,-rpath,$LLVM_PREFIX/lib;-DCMAKE_MODULE_LINKER_FLAGS=-L$LLVM_PREFIX/lib -Wl,-rpath,$LLVM_PREFIX/lib"
                else
                    echo "WARNING: Homebrew LLVM not found. Installing..."
                    echo "Run: brew install llvm"
                    echo ""
                    echo "macOS System Integrity Protection (SIP) blocks system clang's TSan runtime."
                    echo "You need Homebrew LLVM for Thread Sanitizer to work on macOS."
                    exit 1
                fi
            else
                echo "ERROR: Homebrew not found. Thread Sanitizer on macOS requires Homebrew LLVM."
                echo "Install Homebrew from https://brew.sh/ then run: brew install llvm"
                exit 1
            fi
        else
            export CC=clang
            export CXX=clang++

            pip install --verbose -e .[test] \
                --config-settings=cmake.args="-DENABLE_SANITIZER=thread"
        fi

        echo ""
        echo "Running tests with Thread Sanitizer..."
        export TSAN_OPTIONS="second_deadlock_stack=1 history_size=7"

        echo "→ Running vector environment tests (most critical for threading)..."
        python -m pytest tests/python/test_atari_vector_env.py -v -x

        echo "→ Running all tests..."
        python -m pytest -v
        ;;

    address)
        echo "Building with Address Sanitizer + UBSan..."
        export CC=clang
        export CXX=clang++
        pip install --verbose -e .[test] \
            --config-settings=cmake.args="-DENABLE_SANITIZER=address"

        echo ""
        echo "Running tests with Address Sanitizer..."
        export ASAN_OPTIONS="detect_leaks=1:check_initialization_order=1"
        export UBSAN_OPTIONS="print_stacktrace=1"

        echo "→ Running vector environment tests..."
        python -m pytest tests/python/test_atari_vector_env.py -v -x

        echo "→ Running all tests..."
        python -m pytest -v
        ;;

    valgrind-memcheck)
        echo "Building with debug symbols..."
        pip install --verbose -e .[test] \
            --config-settings=cmake.args="-DCMAKE_BUILD_TYPE=RelWithDebInfo"

        echo ""
        echo "Running Valgrind Memcheck (memory leak detection)..."

        if [ ! -f ".valgrind-python.supp" ]; then
            echo "Warning: .valgrind-python.supp not found. Some false positives may appear."
        fi

        valgrind \
            --tool=memcheck \
            --leak-check=full \
            --show-leak-kinds=definite,possible \
            --track-origins=yes \
            --verbose \
            --suppressions=.valgrind-python.supp \
            python -m pytest tests/python/test_atari_vector_env.py::TestVectorEnv::test_reset_step_shapes -v -k "num_envs-1"
        ;;

    valgrind-helgrind)
        echo "Building with debug symbols..."
        pip install --verbose -e .[test] \
            --config-settings=cmake.args="-DCMAKE_BUILD_TYPE=RelWithDebInfo"

        echo ""
        echo "Running Valgrind Helgrind (thread error detection)..."

        if [ ! -f ".valgrind-python.supp" ]; then
            echo "Warning: .valgrind-python.supp not found. Some false positives may appear."
        fi

        valgrind \
            --tool=helgrind \
            --verbose \
            --suppressions=.valgrind-python.supp \
            python -m pytest tests/python/test_atari_vector_env.py::TestVectorEnv::test_batch_size_async -v
        ;;

    *)
        echo "Unknown sanitizer: $SANITIZER"
        echo ""
        echo "Usage: $0 [thread|address|valgrind-memcheck|valgrind-helgrind]"
        echo ""
        echo "Options:"
        echo "  thread              - Thread Sanitizer (detects data races, deadlocks)"
        echo "  address             - Address Sanitizer + UBSan (detects memory errors, UB)"
        echo "  valgrind-memcheck   - Valgrind Memcheck (detects memory leaks)"
        echo "  valgrind-helgrind   - Valgrind Helgrind (detects thread errors)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Sanitizer run complete: $SANITIZER"
echo "=========================================="
