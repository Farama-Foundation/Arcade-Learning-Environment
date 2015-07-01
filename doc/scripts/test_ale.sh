#!/usr/bin/env bash
set +e

if [ $# -lt 1 ]; then
    echo "usage: $0 ROM"
    exit
fi
ROM=$1
LOG="log.txt"
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

function LOG {
    echo "$1"
    echo "$1" >> $LOG
}

# TEST (string command)
function TEST {
    LOG "==> TEST: \"$1\""
    $1 >>$LOG 2>&1
    RETVAL=$?
    if [ $RETVAL -ne 0 ]; then
        LOG "==> FAILURE!!!"
        FAILED_TESTS=$((FAILED_TESTS+1))
    else
        PASSED_TESTS=$((PASSED_TESTS+1))
    fi
}

# TEST_CMAKE_BUILD (Bool USE_SDL, Bool USE_RLGLUE)
function TEST_CMAKE_BUILD {
    echo "=========== [ CMAKE BUILD TESTS ] ==========="
    LOG "USE_SDL = $1; USE_RLGLUE = $2;"
    TEST "cmake -DUSE_SDL=$1 -DUSE_RLGLUE=$2"
    TEST "make clean"
    TEST "make -j16"
}

# TEST_MAKEFILE_BUILD (Bool USE_SDL, Bool USE_RLGLUE)
function TEST_MAKEFILE_BUILD {
    LOG "=========== [ MAKEFILE BUILD TESTS ] ==========="
    LOG "USE_SDL = $1; USE_RLGLUE = $2;"
    ORIG="USE_SDL\ *:= 0"
    REPL="USE_SDL := $1"
    sed -e "s/$ORIG/$REPL/" makefile.$suffix > makefile.tmp
    ORIG="USE_RLGLUE\ *:= 0"
    REPL="USE_RLGLUE := $2"
    
    # Needed because Mac OS X's sed requires a backup extension 
    if [ "$suffix" = "mac" ]; then
        sed -i '' "s/$ORIG/$REPL/" makefile.tmp
    else
        sed -i "s/$ORIG/$REPL/" makefile.tmp
    fi

    TEST "make -f makefile.tmp clean"
    TEST "make -f makefile.tmp -j16"
    rm makefile.tmp
}

# TEST_SHARED_LIBRARY_EXAMPLE (Bool USE_SDL)
function TEST_SHARED_LIBRARY_EXAMPLE {
    cd doc/examples
    LOG "=========== [ SHARED LIBRARY TESTS ] ==========="
    ORIG="USE_SDL\ *:= 0"
    REPL="USE_SDL := $1"
    sed -e "s/$ORIG/$REPL/" Makefile.sharedlibrary > makefile.tmp
    TEST "make -f makefile.tmp clean"
    TEST "make -f makefile.tmp"
    rm makefile.tmp
    if [ -f "sharedLibraryInterfaceExample" ]; then
        export DYLD_LIBRARY_PATH="../..:."
        TEST "./sharedLibraryInterfaceExample $ROM"
    else
        LOG "Skipping TEST: ./sharedLibraryInterfaceExample"
        SKIPPED_TESTS=$((SKIPPED_TESTS+1))
    fi
    cd ../..
    cat doc/examples/$LOG >> $LOG
    rm doc/examples/$LOG
}

# TEST_RLGLUE_EXAMPLE
function TEST_RLGLUE_EXAMPLE {
    LOG "=========== [ RL-GLUE TESTS ] ==========="
    cd doc/examples
    TEST "make -f Makefile.rlglue clean"
    TEST "make -f Makefile.rlglue"
    if [[ -f "RLGlueExperiment" && -f "RLGlueAgent" && -f "../../ale" ]]; then
        rl_glue &> /dev/null &
        ../../ale -game_controller rlglue $ROM &> /dev/null &
        ./RLGlueAgent &> /dev/null &
        TEST "./RLGlueExperiment"
    else
        LOG "Skipping TEST: ./RLGlueExperiment"
        SKIPPED_TESTS=$((SKIPPED_TESTS+1))
    fi
    cd ../..
    cat doc/examples/$LOG >> $LOG
    rm doc/examples/$LOG
}

# TEST_PYTHON_EXAMPLE (Bool USE_SDL)
function TEST_PYTHON_EXAMPLE {
    LOG "=========== [ PYTHON EXAMPLE TESTS ] ==========="
    INSTALLED=`pip list | grep ale-python-interface`
    if [ -z "$INSTALLED" ]; then
        LOG "==> ale-python-interface not installed. Skipping this test..."
        SKIPPED_TESTS=$((SKIPPED_TESTS+1))
    else
        cd doc/examples
        if [ $1 -eq 1 ]; then
            mv python_example.py tmp.py
            ORIG="USE_SDL = False"
            REPL="USE_SDL = True"
            sed -e "s/$ORIG/$REPL/" tmp.py > python_example.py
            TEST "python python_example.py $ROM"
            mv tmp.py python_example.py
        else
            TEST "python python_example.py $ROM"
        fi
        cd ../..
        cat doc/examples/$LOG >> $LOG
        rm doc/examples/$LOG
    fi
}

unamestr=`uname -s`
echo `uname -a` >> $LOG
if [[ "$unamestr" == 'Linux' ]]; then
    suffix="unix"
elif [[ "$unamestr" == 'Darwin' ]]; then
    suffix="mac"
else
    echo "Unknown platform: $unamestr"
    exit
fi

# Makefile Test without SDL or RL_Glue
USE_SDL=0
USE_RLGLUE=0
TEST_MAKEFILE_BUILD $USE_SDL $USE_RLGLUE
TEST_SHARED_LIBRARY_EXAMPLE $USE_SDL

# Makefile Test with SDL and RL_Glue
USE_SDL=1
USE_RLGLUE=1
TEST_MAKEFILE_BUILD $USE_SDL $USE_RLGLUE
TEST_SHARED_LIBRARY_EXAMPLE $USE_SDL
TEST_RLGLUE_EXAMPLE

# CMake with no SDL/RL_Glue
USE_SDL=0
USE_RLGLUE=0
TEST_CMAKE_BUILD $USE_SDL $USE_RLGLUE
TEST_SHARED_LIBRARY_EXAMPLE $USE_SDL

# CMake with SDL/RL_Glue
USE_SDL=1
USE_RLGLUE=1
TEST_CMAKE_BUILD $USE_SDL $USE_RLGLUE
TEST_SHARED_LIBRARY_EXAMPLE $USE_SDL
TEST_RLGLUE_EXAMPLE

# Test the python examples
USE_SDL=0
TEST_PYTHON_EXAMPLE $USE_SDL
USE_SDL=1
TEST_PYTHON_EXAMPLE $USE_SDL

LOG "==> All Tests Finished"
LOG "==> Tests Passed: $PASSED_TESTS"
LOG "==> Tests Failed: $FAILED_TESTS"
LOG "==> Tests Skipped: $SKIPPED_TESTS"
LOG "==> See $LOG for output"
