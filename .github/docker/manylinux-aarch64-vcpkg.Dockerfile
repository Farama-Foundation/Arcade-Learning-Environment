FROM quay.io/pypa/manylinux2014_aarch64

LABEL org.opencontainers.image.source=https://github.com/Farama-Foundation/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar glibc-static gcc gcc-c++ git cmake make

RUN git clone https://github.com/ninja-build/ninja.git /tmp/ninja && \
    cd /tmp/ninja && \
    git checkout v1.12.1 && \
    cmake -Bbuild-cmake -S. -DBUILD_TESTING=OFF && \
    cmake --build build-cmake && \
    cp build-cmake/ninja /usr/local/bin/ && \
    chmod +x /usr/local/bin/ninja && \
    ninja --version && \
    rm -rf /tmp/ninja

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
RUN cd /opt/vcpkg && git reset --hard 9b75e789ece3f942159b8500584e35aafe3979ff

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN mkdir -p /root/.vcpkg && touch /root/.vcpkg/vcpkg.path.txt

RUN bootstrap-vcpkg.sh &&  \
    vcpkg integrate install && \
    vcpkg integrate bash
