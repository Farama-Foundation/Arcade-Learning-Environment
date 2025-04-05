FROM quay.io/pypa/manylinux2014_x86_64

LABEL org.opencontainers.image.source=https://github.com/Farama-Foundation/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar

# Install a specific version of CMake from source
RUN rm -rf /usr/local/bin/cmake || true
RUN which cmake || echo "No cmake found in PATH"

# Install a newer version of Ninja build system
RUN curl -L -o ninja-linux.zip https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip && \
    unzip ninja-linux.zip -d /usr/local/bin && \
    rm ninja-linux.zip && \
    chmod +x /usr/local/bin/ninja && \
    ninja --version

# Debug Ninja installation
RUN ls -la /usr/local/bin/ninja && \
    file /usr/local/bin/ninja && \
    ldd /usr/local/bin/ninja || echo "Not a dynamic executable" && \
    /usr/local/bin/ninja --version

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
RUN cd /opt/vcpkg && git reset --hard 9b75e789ece3f942159b8500584e35aafe3979ff

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN mkdir -p /root/.vcpkg && touch /root/.vcpkg/vcpkg.path.txt

RUN bootstrap-vcpkg.sh &&  \
    vcpkg integrate install && \
    vcpkg integrate bash
