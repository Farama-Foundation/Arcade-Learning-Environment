FROM quay.io/pypa/manylinux2014_x86_64

LABEL org.opencontainers.image.source https://github.com/mgbellemare/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN bootstrap-vcpkg.sh && \
    vcpkg integrate install && \
    vcpkg integrate bash
