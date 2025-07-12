# WhisperX API Server for RunPod

A production-ready, GPU-accelerated WhisperX API server that provides OpenAI Whisper API compatible endpoints with advanced features like speaker diarization and word-level alignment.

## Features

- 🎯 **OpenAI Whisper API Compatible** - Drop-in replacement for OpenAI's Whisper API
- 🔊 **Speaker Diarization** - Identify different speakers in audio
- 📝 **Word-Level Alignment** - Precise word timestamps and positioning
- 🚀 **GPU Acceleration** - Optimized for NVIDIA GPUs with CUDA support
- 🏗️ **Scalable Architecture** - Ready for serverless deployment on RunPod
- 📊 **Multiple Output Formats** - JSON, SRT, VTT subtitle formats
- 🌐 **Multi-Language Support** - Support for 90+ languages
- 🔄 **Model Caching** - Efficient model loading and memory management

## API Endpoints

### Core Endpoints

- `POST /v1/audio/transcriptions` - Transcribe audio with optional diarization and alignment
- `GET /healthcheck` - Health check endpoint
- `GET /models/list` - List available models
- `GET /` - Root endpoint with API information

## Quick Start with RunPod

### 1. Repository Setup

1. **Fork/Clone this repository** to your GitHub account
2. **Ensure your repository is public** (required for RunPod GitHub integration)
3. **Verify the `Dockerfile` is in the root directory**

### 2. Deploy to RunPod

1. **Connect GitHub to RunPod**
   - Go to RunPod Dashboard → Settings → Connections
   - Authorize GitHub access

2. **Create New Serverless Endpoint**
   - Navigate to Serverless → New Endpoint
   - Select "Custom Source" → "GitHub Repo"
   - Choose your repository and branch
   - Set container port to `8000`

3. **Configure Resources**
   - **Recommended GPU**: NVIDIA T4, RTX 3090, or A10G
   - **Minimum VRAM**: 8GB (12GB+ recommended for large models)
   - **Disk Size**: 50GB minimum
   - **Memory**: 16GB recommended

4. **Environment Variables** (Optional)
   - `WHISPER_MODEL`: Model to load (default: "large-v2")
   - `HF_TOKEN`: Hugging Face token for diarization (required for speaker diarization)
   - `CUDA_VISIBLE_DEVICES`: GPU device to use (default: "0")

5. **Deploy**
   - Click "Create Endpoint"
   - Wait for build completion (typically 5-10 minutes)
   - Get your endpoint URL: `https://<endpoint-id>-8000.proxy.runpod.net`

## Usage Examples

### Basic Transcription

```bash
curl -X POST "https://your-endpoint-url/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=whisper-large-v2"
```

### With Speaker Diarization

```bash
curl -X POST "https://your-endpoint-url/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

### With Word-Level Alignment

```bash
curl -X POST "https://your-endpoint-url/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "align=true" \
  -F "timestamp_granularities=word"
```

## Supported Models

- `whisper-large-v2` - Best accuracy, slower inference
- `whisper-large-v3` - Latest model with improved performance
- `whisper-medium` - Good balance of speed and accuracy
- `whisper-small` - Fastest inference, lower accuracy

## Troubleshooting

### Common Issues

1. **Build Fails**
   - Check Dockerfile path and syntax
   - Verify all dependencies in requirements.txt
   - Ensure base image supports CUDA

2. **API Not Responding**
   - Verify port 8000 is exposed
   - Check container logs for errors
   - Ensure sufficient GPU memory

3. **Diarization Not Working**
   - Set `HF_TOKEN` environment variable
   - Verify Hugging Face token has access to pyannote models
   - Check model loading logs

## Local Development

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- 8GB+ GPU memory

### Setup

```bash
# Clone repository
git clone https://github.com/703deuce/whisper.git
cd whisper

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_hugging_face_token
export WHISPER_MODEL=large-v2

# Run server
python app.py
```

## License

This project is licensed under the MIT License. 