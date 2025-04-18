FROM quay.io/pypa/manylinux2014_x86_64
LABEL org.opencontainers.image.source=https://github.com/Farama-Foundation/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar wget openssl-devel bzip2-devel libffi-devel

# Install Python 3.10 from source
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz && \
    tar xzf Python-3.10.13.tgz && \
    cd Python-3.10.13 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    rm -rf /tmp/Python-3.10.13*

# Create symbolic links for Python 3.10
RUN ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip

# Install a specific version of CMake from source
RUN rm -rf /usr/local/bin/cmake || true
RUN which cmake || echo "No cmake found in PATH"
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.30.6/cmake-3.30.6-linux-x86_64.sh -o cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr && \
    rm cmake.sh
RUN which cmake && cmake --version

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

# Install package
RUN ls /usr/local
COPY . /usr/local/Arcade-Learning-Environment/
WORKDIR /usr/local/Arcade-Learning-Environment/
