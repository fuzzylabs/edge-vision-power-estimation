FROM nvcr.io/nvidia/pytorch:24.06-py3-igpu

# Create a user group
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

USER appuser

WORKDIR /app

COPY --chown=appuser:appuser . ./

RUN apt update && apt install --no-install-recommends -y libgl1-mesa-glx

RUN pip install ultralytics
