#!/bin/bash
set -e

# Image name
IMAGE_NAME="lpggen:latest"

# Build the Docker image
echo "Building Docker image ${IMAGE_NAME}..."
docker build --platform=linux/amd64 -t ${IMAGE_NAME} .

# Create output directory if it doesn't exist
mkdir -p ./output

# Run the container
# We mount the local ./output directory to /work/output inside the container
# Arguments passed to this script are forwarded to generate_dataset.py
echo "Running container..."
docker run --rm --platform=linux/amd64 \
  -v "$(pwd)/output:/work/output" \
  ${IMAGE_NAME} \
  python3 generate_dataset.py "$@"
