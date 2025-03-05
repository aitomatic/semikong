FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl=7.81.0-1ubuntu1.20 ca-certificates=20240203~22.04.1 git=1:2.34.1-1ubuntu1.12 vim=2:8.2.3995-1ubuntu2.23 && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/opt/conda/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT="/opt/conda"
ENV UV_SYSTEM_PYTHON=true

WORKDIR /workspace
