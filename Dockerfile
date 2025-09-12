# Use official CUDA image as base. Override with `--build-arg BASE_IMAGE=...`
ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
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
RUN python -m pip install --upgrade pip setuptools

# Install t5x (patched for Python 3.10 compatibility)
# t5x installs latest flax, but latest flax requires Python >3.10
# Two options: (1) upgrade Python, or (2) patch t5x to use flax 0.10.6
# Here we settle for (2), as (1) led to an endless chain of dependency issues
# TODO(chrisdonahue): Support newer versions of Python
RUN git clone https://github.com/google-research/t5x.git /t5x && \
    pushd /t5x && \
    git checkout 92c5b467a5964d06c351c7eae4aa4bcd341c7ded && \
    sed -i 's|flax @ git+https://github.com/google/flax#egg=flax|flax==0.10.6|g' setup.py && \
    python -m pip install -e .[gpu] && \
    popd

# Create Magenta RealTime library placeholder
ENV MAGENTA_RT_CACHE_DIR=/magenta-realtime/cache
ENV MAGENTA_RT_LIB_DIR=/magenta-realtime/magenta_rt
RUN mkdir -p $MAGENTA_RT_CACHE_DIR
RUN mkdir -p $MAGENTA_RT_LIB_DIR
COPY setup.py .
COPY pyproject.toml .
RUN sed -i 's|t5x\[gpu\] @ git+https://github.com/google-research/t5x\.git@92c5b46|t5x[gpu]|g' pyproject.toml && \
    sed -i 's|t5x @ git+https://github.com/google-research/t5x\.git@92c5b46|t5x|g' pyproject.toml

# Install Magenta RealTime and dependencies
RUN python -m pip install -e .[gpu]
RUN python -m pip uninstall -y tensorflow tensorflow-cpu tensorflow-text
RUN python -m pip install tf-nightly==2.20.0.dev20250619 tensorflow-text-nightly==2.20.0.dev20250316

# Copy library and tests (last, to improve caching)
COPY magenta_rt magenta_rt
COPY test test

CMD ["python", "-m", "magenta_rt.generate"]
