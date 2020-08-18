#!/bin/bash
set -e -u -x


function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}


yum install -y wget
yum install -y zlib-devel

CMAKE_DIR="/cmk"
mkdir ${CMAKE_DIR} && cd ${CMAKE_DIR}
wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3-Linux-x86_64.tar.gz
tar -xvf cmake-3.17.3-Linux-x86_64.tar.gz > /dev/null
mv cmake-3.17.3-Linux-x86_64 cmake-install
PATH=${CMAKE_DIR}/cmake-install:${CMAKE_DIR}/cmake-install/bin:$PATH
cd ..

for PYBIN in /opt/python/*/bin; do
    if [[ ( "$PYBIN" == *"36"* ) || ( "$PYBIN" == *"37"* ) || ( "$PYBIN" == *"38"* ) ]]; then
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
    fi
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done
