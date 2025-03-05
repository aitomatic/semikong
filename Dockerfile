FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git vim && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/opt/conda/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT="/opt/conda"
ENV UV_SYSTEM_PYTHON=true

WORKDIR /workspace
