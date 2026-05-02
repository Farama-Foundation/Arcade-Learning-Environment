#!/bin/bash
# Helper script for running sanitizers locally and in CI
# Usage: ./scripts/run_sanitizers.sh [thread|address|valgrind-memcheck|valgrind-helgrind] [quick|full]
#
# Test scope (default: quick):
#   quick - Run a small focused set of tests (fast; default for local dev and CI)
#   full  - Run the full test suite (slow; opt-in)

set -e

SANITIZER=${1:-thread}
TEST_SCOPE=${2:-quick}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ "$TEST_SCOPE" != "quick" && "$TEST_SCOPE" != "full" ]]; then
    echo "Error: Invalid test scope '$TEST_SCOPE'. Must be 'quick' or 'full'."
    exit 1
fi

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running sanitizer: $SANITIZER (scope: $TEST_SCOPE)"
echo "=========================================="

# Ensure ROMs are downloaded
if [ ! -d "src/ale/python/roms" ]; then
    echo "Downloading ROMs..."
    ./scripts/download_unpack_roms.sh
fi

# Focused test selection used by both TSan and ASan when scope=quick.
# Targets the threading hot paths in the vectorized env.
QUICK_KEYS="test_batch_size_async or test_episodic_life_equivalence or (test_reset_step_shapes and 4-3-ALE)"

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
            # Honor CC/CXX from the environment so CI (which sets clang-18) is respected.
            export CC=${CC:-clang}
            export CXX=${CXX:-clang++}

            pip install --verbose -e .[test] \
                --config-settings=cmake.define.ENABLE_SANITIZER=thread
        fi

        echo ""
        echo "Running tests with Thread Sanitizer..."
        export TSAN_OPTIONS="second_deadlock_stack=1 history_size=7 halt_on_error=1"

        # Python isn't built with TSan, so the instrumented .so has unresolved
        # TSan runtime symbols when imported. Preload the runtime to provide them.
        TSAN_RT=$("${CXX:-clang++}" -print-file-name=libclang_rt.tsan-x86_64.so 2>/dev/null)
        if [ -f "$TSAN_RT" ]; then
            export LD_PRELOAD="$TSAN_RT${LD_PRELOAD:+:$LD_PRELOAD}"
            echo "Preloading TSan runtime: $TSAN_RT"
        else
            echo "WARNING: TSan runtime not found via '${CXX:-clang++} -print-file-name'."
            echo "Tests will likely fail with 'undefined symbol: __tsan_*' errors."
        fi

        if [ "$TEST_SCOPE" = "full" ]; then
            echo "→ Running vector environment tests..."
            python -m pytest tests/python/test_atari_vector_env.py -v -x
            echo "→ Running all tests..."
            python -m pytest -v
        else
            echo "→ Running focused vector environment tests (threading hot paths)..."
            python -m pytest tests/python/test_atari_vector_env.py -v -x -s \
                -k "$QUICK_KEYS"
        fi
        ;;

    address)
        echo "Building with Address Sanitizer + UBSan..."
        # Honor CC/CXX from the environment so CI (which sets clang-18) is respected.
        export CC=${CC:-clang}
        export CXX=${CXX:-clang++}
        pip install --verbose -e .[test] \
            --config-settings=cmake.define.ENABLE_SANITIZER=address

        echo ""
        echo "Running tests with Address Sanitizer..."
        export ASAN_OPTIONS="detect_leaks=1:check_initialization_order=1:halt_on_error=1"
        export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"

        # Suppress leak reports from Python/pybind11 module-init machinery
        # (type objects, interpreter-lifetime allocations). See scripts/lsan-suppressions.txt.
        if [ -f "$SCRIPT_DIR/lsan-suppressions.txt" ]; then
            export LSAN_OPTIONS="suppressions=$SCRIPT_DIR/lsan-suppressions.txt:print_suppressions=0"
            echo "Using LSan suppressions: $SCRIPT_DIR/lsan-suppressions.txt"
        fi

        # Python isn't built with ASan, so preload the runtime to satisfy any
        # symbols not auto-loaded via DT_NEEDED on the instrumented .so.
        ASAN_RT=$("${CXX:-clang++}" -print-file-name=libclang_rt.asan-x86_64.so 2>/dev/null)
        if [ -f "$ASAN_RT" ]; then
            export LD_PRELOAD="$ASAN_RT${LD_PRELOAD:+:$LD_PRELOAD}"
            echo "Preloading ASan runtime: $ASAN_RT"
        else
            echo "WARNING: ASan runtime not found via '${CXX:-clang++} -print-file-name'."
        fi

        if [ "$TEST_SCOPE" = "full" ]; then
            echo "→ Running vector environment tests..."
            python -m pytest tests/python/test_atari_vector_env.py -v -x
            echo "→ Running all tests..."
            python -m pytest -v
        else
            echo "→ Running focused vector environment tests..."
            python -m pytest tests/python/test_atari_vector_env.py -v -x -s \
                -k "$QUICK_KEYS"
        fi
        ;;

    valgrind-memcheck)
        echo "Building with debug symbols..."
        # Use cmake.build-type (not cmake.define.CMAKE_BUILD_TYPE) — scikit-build-core
        # sets cmake.build-type=Release in pyproject.toml and the define-form is
        # ignored, leaving us with a stripped Release build that valgrind can't
        # symbolize. The CMakeLists guard also disables IPO for RelWithDebInfo.
        pip install --verbose -e .[test] \
            --config-settings=cmake.build-type=RelWithDebInfo

        echo ""
        echo "Running Valgrind Memcheck (memory leak detection)..."

        if [ ! -f "$SCRIPT_DIR/valgrind-python.supp" ]; then
            echo "Warning: $SCRIPT_DIR/valgrind-python.supp not found. Some false positives may appear."
        fi

        if [ "$TEST_SCOPE" = "full" ]; then
            MEMCHECK_TEST_ARGS=(tests/python/test_atari_vector_env.py -v)
        else
            MEMCHECK_TEST_ARGS=(tests/python/test_atari_vector_env.py::TestVectorEnv::test_reset_step_shapes -v -s -k "num_envs-1")
        fi

        valgrind \
            --tool=memcheck \
            --leak-check=full \
            --show-leak-kinds=definite,possible \
            --errors-for-leak-kinds=definite \
            --track-origins=yes \
            --error-exitcode=1 \
            --verbose \
            --suppressions="$SCRIPT_DIR/valgrind-python.supp" \
            python -m pytest "${MEMCHECK_TEST_ARGS[@]}"
        ;;

    valgrind-helgrind)
        echo "Building with debug symbols..."
        # Use cmake.build-type (not cmake.define.CMAKE_BUILD_TYPE) — scikit-build-core
        # sets cmake.build-type=Release in pyproject.toml and the define-form is
        # ignored, leaving us with a stripped Release build that valgrind can't
        # symbolize. The CMakeLists guard also disables IPO for RelWithDebInfo.
        pip install --verbose -e .[test] \
            --config-settings=cmake.build-type=RelWithDebInfo

        echo ""
        echo "Running Valgrind Helgrind (thread error detection)..."

        if [ ! -f "$SCRIPT_DIR/valgrind-python.supp" ]; then
            echo "Warning: $SCRIPT_DIR/valgrind-python.supp not found. Some false positives may appear."
        fi

        if [ "$TEST_SCOPE" = "full" ]; then
            HELGRIND_TEST_ARGS=(tests/python/test_atari_vector_env.py -v)
        else
            HELGRIND_TEST_ARGS=(tests/python/test_atari_vector_env.py::TestVectorEnv::test_batch_size_async -v -s)
        fi

        valgrind \
            --tool=helgrind \
            --error-exitcode=1 \
            --verbose \
            --suppressions="$SCRIPT_DIR/valgrind-python.supp" \
            python -m pytest "${HELGRIND_TEST_ARGS[@]}"
        ;;

    *)
        echo "Unknown sanitizer: $SANITIZER"
        echo ""
        echo "Usage: $0 [thread|address|valgrind-memcheck|valgrind-helgrind] [quick|full]"
        echo ""
        echo "Sanitizers:"
        echo "  thread              - Thread Sanitizer (detects data races, deadlocks)"
        echo "  address             - Address Sanitizer + UBSan (detects memory errors, UB)"
        echo "  valgrind-memcheck   - Valgrind Memcheck (detects memory leaks)"
        echo "  valgrind-helgrind   - Valgrind Helgrind (detects thread errors)"
        echo ""
        echo "Test scope (default: quick):"
        echo "  quick - Run a small set of focused tests (fast)"
        echo "  full  - Run the full test suite (slow; opt-in)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Sanitizer run complete: $SANITIZER (scope: $TEST_SCOPE)"
echo "=========================================="
