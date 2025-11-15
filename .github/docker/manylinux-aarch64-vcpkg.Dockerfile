FROM quay.io/pypa/manylinux2014_aarch64

LABEL org.opencontainers.image.source=https://github.com/Farama-Foundation/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar glibc-static gcc gcc-c++ git cmake make perl

# Install CUDA Toolkit for XLA GPU support (using runfile installer for aarch64)
RUN curl -fsSL -o /tmp/cuda_12.6.2_560.35.03_linux_sbsa.run \
    https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux_sbsa.run && \
    sh /tmp/cuda_12.6.2_560.35.03_linux_sbsa.run --silent --toolkit --no-man-page --no-drm && \
    rm /tmp/cuda_12.6.2_560.35.03_linux_sbsa.run

ENV CUDA_HOME="/usr/local/cuda-12.6"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

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
