FROM quay.io/pypa/manylinux2014_x86_64

LABEL org.opencontainers.image.source=https://github.com/Farama-Foundation/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar glibc-static perl

# Install CUDA Toolkit for XLA GPU support (minimal installation: nvcc + cudart only)
RUN curl -fsSL -o /tmp/cuda_12.6.2_560.35.03_linux.run \
    https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run && \
    sh /tmp/cuda_12.6.2_560.35.03_linux.run --silent --toolkit --no-opengl-libs \
    --no-man-page --no-drm --tmpdir=/tmp/cuda-install && \
    rm -rf /tmp/cuda_12.6.2_560.35.03_linux.run /tmp/cuda-install && \
    rm -rf /usr/local/cuda-12.6/nsight* /usr/local/cuda-12.6/libnvvp

ENV CUDA_HOME="/usr/local/cuda-12.6"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
RUN cd /opt/vcpkg && git reset --hard 9b75e789ece3f942159b8500584e35aafe3979ff

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN mkdir -p /root/.vcpkg && touch /root/.vcpkg/vcpkg.path.txt

RUN bootstrap-vcpkg.sh &&  \
    vcpkg integrate install && \
    vcpkg integrate bash
