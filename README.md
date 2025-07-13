# WhisperX API Server for RunPod

A scalable, GPU-powered transcription API server built with WhisperX and FastAPI, designed for deployment on RunPod's serverless platform.

## Features

- **OpenAI-compatible API** endpoints (`/v1/audio/transcriptions`, `/v1/audio/translations`)
- **Speaker diarization** - Identify who spoke when
- **Word-level timestamps** - Precise timing for each word
- **90+ language support** - Multi-language transcription
- **Multiple output formats** - JSON, SRT, VTT
- **GPU acceleration** - Optimized for NVIDIA GPUs
- **Scalable deployment** - RunPod serverless platform

## API Endpoints

- `POST /v1/audio/transcriptions` - Transcribe audio with diarization and timestamps
- `POST /v1/audio/translations` - Translate audio to English
- `GET /healthcheck` - Service health and status
- `GET /models/list` - Available models

## Quick Start

### Prerequisites

1. **Docker Desktop** - Install from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. **Docker Hub Account** - Create at [https://hub.docker.com](https://hub.docker.com)
3. **RunPod Account** - Sign up at [https://www.runpod.io](https://www.runpod.io)

### Deployment Method 1: Local Build + Docker Hub Push

This method gives you full control over the build process and faster deployment.

#### Step 1: Clone and Build

```bash
git clone https://github.com/703deuce/whisper.git
cd whisper
```

#### Step 2: Build and Push to Docker Hub

**For Windows (PowerShell):**
```powershell
.\build-and-push.ps1 -DockerHubUsername "yourusername"
```

**For Linux/macOS:**
```bash
chmod +x build-and-push.sh
./build-and-push.sh yourusername
```

**Manual build (if scripts don't work):**
```bash
# Build the image
docker build -t yourusername/whisperx-api-server:latest -f Dockerfile.cuda .

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push yourusername/whisperx-api-server:latest
```

#### Step 3: Deploy on RunPod

1. Go to [RunPod Dashboard](https://www.runpod.io/console)
2. Navigate to **Pods** or **Serverless Endpoints**
3. Click **Create New Pod/Endpoint**
4. Configure:
   - **Container Image**: `yourusername/whisperx-api-server:latest`
   - **Port**: `8000`
   - **GPU**: Choose RTX 3090, A10G, or T4 (8-12GB VRAM recommended)
   - **Container Disk**: 20GB minimum
5. Click **Deploy**

#### Step 4: Test Your API

RunPod will provide a proxy URL like `https://<pod-id>-8000.proxy.runpod.net`

```bash
# Test health check
curl "https://<pod-id>-8000.proxy.runpod.net/healthcheck"

# Test transcription
curl -X POST "https://<pod-id>-8000.proxy.runpod.net/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@youraudiofile.wav" \
  -F "diarize=true" \
  -F "align=true" \
  -F "timestamp_granularities=word"
```

### Deployment Method 2: RunPod CLI

Install and configure the RunPod CLI:

```bash
# Install RunPod CLI
pip install runpod

# Deploy
runpodctl create pod \
  --name whisperx-api-server \
  --imageName yourusername/whisperx-api-server:latest \
  --gpuType "NVIDIA RTX 3090" \
  --gpuCount 1 \
  --containerDiskSize 20 \
  --ports "8000/http"
```

## API Usage Examples

### Basic Transcription

```bash
curl -X POST "https://your-pod-url/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

### With Speaker Diarization

```bash
curl -X POST "https://your-pod-url/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "diarize=true" \
  -F "align=true"
```

### Word-level Timestamps

```bash
curl -X POST "https://your-pod-url/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "timestamp_granularities=word"
```

### SRT Format Output

```bash
curl -X POST "https://your-pod-url/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "response_format=srt"
```

## Configuration Options

### Environment Variables

- `WHISPER_MODEL` - Model size (default: "large-v2")
- `DEVICE` - Device to use (default: "cuda")
- `COMPUTE_TYPE` - Compute type (default: "float16")
- `BATCH_SIZE` - Batch size (default: 16)
- `HF_TOKEN` - Hugging Face token for speaker diarization

### Request Parameters

- `file` - Audio file (required)
- `model` - Model to use (default: "whisper-1")
- `language` - Language code (auto-detect if not specified)
- `response_format` - Output format: "json", "srt", "vtt" (default: "json")
- `timestamp_granularities` - "segment" or "word" (default: "segment")
- `diarize` - Enable speaker diarization (default: false)
- `align` - Enable word-level alignment (default: false)

## Performance Tips

1. **GPU Selection**: Use RTX 3090 or A10G for best performance
2. **Model Size**: "large-v2" offers best accuracy, "base" for faster processing
3. **Batch Size**: Adjust based on GPU memory (16 for 12GB+, 8 for 8GB)
4. **Scaling**: Use multiple pods for high-volume processing

## Troubleshooting

### Common Issues

1. **Build failures**: Check Docker installation and Dockerfile.cuda exists
2. **Memory errors**: Use smaller batch size or model
3. **Slow transcription**: Ensure GPU is properly configured
4. **API errors**: Check logs in RunPod dashboard

### Health Check

```bash
curl "https://your-pod-url/healthcheck"
```

Expected response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": true,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Development

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Test health check
python test_health.py
```

## Support

For issues and questions:
- Check RunPod documentation
- Review container logs in RunPod dashboard
- Verify GPU and memory requirements

## License

This project is open source and available under the MIT License. 