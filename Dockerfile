
FROM pytorch/libtorch-cxx11-builder:cuda118 as base

RUN apt update -q && \
    apt install -qq -y \
        software-properties-common \
        libssl-dev \
        libboost-program-options-dev && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update -q && \
    apt install -qq -y \
        gcc-11 \
        g++-11 && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip && \
    unzip -q libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
ENV LIBTORCH_DIR /libtorch

# RUN cp /usr/local/cuda/extras/CUPTI/lib64/libcupti* /usr/local/cuda/lib64s

COPY . /app

WORKDIR /app/build
RUN cmake .. -DCMAKE_PREFIX_PATH=${LIBTORCH_DIR} && \
    make


