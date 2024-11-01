FROM nvcr.io/nvidia/pytorch:24.09-py3

RUN pip install 'git+https://github.com/NVIDIA/TensorRT.git/#egg=trex&subdirectory=tools/experimental/trt-engine-explorer'

# Create a user group
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

USER appuser

WORKDIR /app

COPY --chown=appuser:appuser . ./