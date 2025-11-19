FROM quay.io/pypa/manylinux_2_28_aarch64

LABEL org.opencontainers.image.source=https://github.com/Farama-Foundation/Arcade-Learning-Environment

RUN dnf install -y git cmake make curl unzip zip tar \
    gcc gcc-c++ dnf-plugins-core ninja-build glibc-static

RUN dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo

RUN dnf install -y \
    cuda-minimal-build-12-6 \
    cuda-cudart-devel-12-6 \
    && dnf clean all

ENV CUDA_HOME="/usr/local/cuda-12.6"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg && \
    cd /opt/vcpkg && \
    git reset --hard 9b75e789ece3f942159b8500584e35aafe3979ff

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN mkdir -p /root/.vcpkg && touch /root/.vcpkg/vcpkg.path.txt

RUN bootstrap-vcpkg.sh &&  \
    vcpkg integrate install && \
    vcpkg integrate bash
