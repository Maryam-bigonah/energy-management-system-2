FROM mcr.microsoft.com/dotnet/runtime:8.0

# Install system dependencies required by LPG binaries and Python packages
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    libc6 \
    libstdc++6 \
    libgomp1 \
    unzip \
    procps \
    libicu-dev \
    libfontconfig1 \
    libfreetype6 \
    libx11-6 \
    libice6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgdiplus \
    libgl1 \
    libssl-dev \
    openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the content of the current directory (host's /work/load) into the container
COPY . /app

# Install pylpg from local source in editable mode
RUN pip install -e /app/pylpg-main

# Setup LPG binaries
# Pylpg expects the 'LPG_linux' folder to be adjacent to 'lpg_execution.py' inside the package.
# We copy the provided binaries to that location and ensure execution permissions.
RUN mkdir -p /app/pylpg-main/pylpg/LPG_linux && \
    cp -r /app/LPG10.10.0_linux/* /app/pylpg-main/pylpg/LPG_linux/ && \
    chmod +x /app/pylpg-main/pylpg/LPG_linux/simengine2

# Default command runs the dataset generator
CMD ["python", "generate_dataset.py"]
