FROM quay.io/pypa/manylinux2014_aarch64

LABEL org.opencontainers.image.source=https://github.com/Farama-Foundation/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar glibc-static gcc

# Install ninja from source to avoid glibc compatibility issues
RUN cd /tmp && \
    git clone --depth 1 --branch v1.11.1 https://github.com/ninja-build/ninja.git && \
    cd ninja && \
    python3 configure.py --bootstrap && \
    cp ninja /usr/local/bin/ && \
    cd / && rm -rf /tmp/ninja

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
RUN cd /opt/vcpkg && git reset --hard 9b75e789ece3f942159b8500584e35aafe3979ff

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN mkdir -p /root/.vcpkg && touch /root/.vcpkg/vcpkg.path.txt

RUN bootstrap-vcpkg.sh &&  \
    vcpkg integrate install && \
    vcpkg integrate bash
