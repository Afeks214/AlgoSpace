#!/bin/bash
# Start AlgoSpace Development Environment

echo "Starting AlgoSpace Development Environment..."

# Check if Docker is in Linux mode
if ! docker info 2>&1 | grep -q "OSType: linux"; then
    echo "ERROR: Docker is not in Linux mode!"
    echo "Please switch Docker to Linux containers mode and try again."
    exit 1
fi

# Check if image exists
if ! docker images | grep -q "algospace-env"; then
    echo "Building Docker image..."
    docker build -t algospace-env -f Dockerfile.light . || {
        echo "Failed to build image. Trying with main Dockerfile..."
        docker build -t algospace-env .
    }
fi

# Run container with interactive shell
echo "Launching container..."
docker run -it --rm \
    -v "$(pwd):/app" \
    -w /app \
    -p 8000:8000 \
    -p 8888:8888 \
    --name algospace-dev \
    algospace-env bash

echo "Development session ended."