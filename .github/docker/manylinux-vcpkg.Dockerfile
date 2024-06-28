FROM quay.io/pypa/manylinux2014_x86_64

LABEL org.opencontainers.image.source https://github.com/Farama-Foundation/Arcade-Learning-Environment

RUN yum install -y curl unzip zip tar pkgconfig

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
RUN cd vcpkg
RUN git reset --hard 8150939b69720adc475461978e07c2d2bf5fb76e
RUN cd ..

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN mkdir -p /root/.vcpkg && touch /root/.vcpkg/vcpkg.path.txt

RUN bootstrap-vcpkg.sh &&  \
    vcpkg integrate install && \
    vcpkg integrate bash
