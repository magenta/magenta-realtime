# Use official CUDA image as base. Override with `--build-arg BASE_IMAGE=...`
ARG BASE_IMAGE=nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

# Configure shell
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Change workdir
WORKDIR /magenta-realtime

# Install core deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python -m pip install --upgrade pip setuptools --ignore-installed
RUN python --version

# Install t5x (patched for Python 3.12 compatibility)
COPY patch/t5x_setup.py.patch patch/t5x_setup.py.patch
RUN git clone https://github.com/google-research/t5x.git /t5x && \
    pushd /t5x && \
    git checkout 7781d167ab421dae96281860c09d5bd785983853 && \
    patch setup.py < /magenta-realtime/patch/t5x_setup.py.patch && \
    python -m pip install .[gpu] && \
    popd

# Create Magenta RealTime library placeholder
ENV MAGENTA_RT_CACHE_DIR=/magenta-realtime/cache
ENV MAGENTA_RT_LIB_DIR=/magenta-realtime/magenta_rt
RUN mkdir -p $MAGENTA_RT_CACHE_DIR
RUN mkdir -p $MAGENTA_RT_LIB_DIR

# Install Magenta RealTime and dependencies
COPY setup.py .
COPY pyproject.toml .
RUN python -m pip install -e .[gpu]
RUN python -m pip install tf2jax==0.3.8

# Apply patches
COPY patch patch
RUN patch /usr/local/lib/python3.12/dist-packages/t5x/partitioning.py < patch/t5x_partitioning.py.patch
RUN patch /usr/local/lib/python3.12/dist-packages/seqio/vocabularies.py < patch/seqio_vocabularies.py.patch

# Copy library and tests (last, to improve caching)
COPY magenta_rt magenta_rt
COPY test test

CMD ["python", "-m", "magenta_rt.generate"]
