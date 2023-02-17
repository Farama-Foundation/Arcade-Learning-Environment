group "default" {
    targets = []
}

variable "MANYLINUX_VERSION" {
    default = "2014"
}

variable "MANYLINUX_ARCH" {
    default = "x86_64"
}

target "manylinux-vcpkg" {
    tags = ["ghcr.io/jessefarebro/manylinux${MANYLINUX_VERSION}_${MANYLINUX_ARCH}-vcpkg"]
    dockerfile = "manylinux-vcpkg.Dockerfile"
    args = {
        BASE_IMAGE = "quay.io/pypa/manylinux${MANYLINUX_VERSION}_${MANYLINUX_ARCH}"
    }
}
