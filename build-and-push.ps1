#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build and push WhisperX API server Docker image to Docker Hub
.DESCRIPTION
    This script builds the Docker image locally and pushes it to Docker Hub for RunPod deployment
.PARAMETER DockerHubUsername
    Your Docker Hub username
.PARAMETER Tag
    Image tag (default: latest)
.EXAMPLE
    .\build-and-push.ps1 -DockerHubUsername "yourusername"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$Tag = "latest"
)

# Configuration
$ImageName = "$DockerHubUsername/whisperx-api-server:$Tag"
$DockerfilePath = "Dockerfile.cuda"

Write-Host "🐳 Building WhisperX API Server Docker Image..." -ForegroundColor Green
Write-Host "Image: $ImageName" -ForegroundColor Yellow
Write-Host "Dockerfile: $DockerfilePath" -ForegroundColor Yellow

# Check if Docker is installed
try {
    docker --version | Out-Null
    Write-Host "✅ Docker is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check if Dockerfile exists
if (!(Test-Path $DockerfilePath)) {
    Write-Host "❌ $DockerfilePath not found" -ForegroundColor Red
    exit 1
}

# Build the Docker image
Write-Host "🔨 Building Docker image..." -ForegroundColor Blue
docker build -t $ImageName -f $DockerfilePath .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker image built successfully!" -ForegroundColor Green

# Login to Docker Hub
Write-Host "🔑 Logging into Docker Hub..." -ForegroundColor Blue
docker login

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker Hub login failed" -ForegroundColor Red
    exit 1
}

# Push the image
Write-Host "⬆️ Pushing image to Docker Hub..." -ForegroundColor Blue
docker push $ImageName

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker push failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Image pushed successfully!" -ForegroundColor Green
Write-Host "🎉 Your image is now available at: $ImageName" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Next steps for RunPod deployment:" -ForegroundColor Yellow
Write-Host "1. Go to RunPod dashboard" -ForegroundColor White
Write-Host "2. Create a new Pod/Serverless Endpoint" -ForegroundColor White
Write-Host "3. Use image: $ImageName" -ForegroundColor White
Write-Host "4. Set port: 8000" -ForegroundColor White
Write-Host "5. Choose GPU with 8-12GB VRAM (RTX 3090, A10G, etc.)" -ForegroundColor White 