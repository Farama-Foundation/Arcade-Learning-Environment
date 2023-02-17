ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source https://github.com/mgbellemare/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN mkdir -p /root/.vcpkg && touch /root/.vcpkg/vcpkg.path.txt

RUN bootstrap-vcpkg.sh &&  \
    vcpkg integrate install && \
    vcpkg integrate bash
