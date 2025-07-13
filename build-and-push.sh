#!/bin/bash

# WhisperX API Server - Build and Push to Docker Hub
# Usage: ./build-and-push.sh <dockerhub-username> [tag]

set -e

# Configuration
DOCKERHUB_USERNAME=${1:-}
TAG=${2:-latest}

if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "❌ Usage: $0 <dockerhub-username> [tag]"
    echo "   Example: $0 yourusername latest"
    exit 1
fi

IMAGE_NAME="$DOCKERHUB_USERNAME/whisperx-api-server:$TAG"
DOCKERFILE_PATH="Dockerfile.cuda"

echo "🐳 Building WhisperX API Server Docker Image..."
echo "Image: $IMAGE_NAME"
echo "Dockerfile: $DOCKERFILE_PATH"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    echo "Please install Docker from https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker is installed"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "❌ $DOCKERFILE_PATH not found"
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .

echo "✅ Docker image built successfully!"

# Login to Docker Hub
echo "🔑 Logging into Docker Hub..."
docker login

# Push the image
echo "⬆️ Pushing image to Docker Hub..."
docker push "$IMAGE_NAME"

echo "✅ Image pushed successfully!"
echo "🎉 Your image is now available at: $IMAGE_NAME"
echo ""
echo "🚀 Next steps for RunPod deployment:"
echo "1. Go to RunPod dashboard"
echo "2. Create a new Pod/Serverless Endpoint"
echo "3. Use image: $IMAGE_NAME"
echo "4. Set port: 8000"
echo "5. Choose GPU with 8-12GB VRAM (RTX 3090, A10G, etc.)" 