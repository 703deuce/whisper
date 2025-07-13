# WhisperX Serverless API for RunPod

A scalable, GPU-powered transcription serverless function built with WhisperX, designed for deployment on RunPod's serverless platform.

## Features

- **🚀 Serverless Architecture** - Auto-scaling, pay-per-use function
- **🎯 WhisperX Integration** - Advanced transcription with speaker diarization
- **🔊 Speaker Diarization** - Identify who spoke when
- **📝 Word-Level Timestamps** - Precise timing for each word
- **🌐 90+ Language Support** - Multi-language transcription
- **📊 Multiple Output Formats** - JSON, SRT, VTT subtitle formats
- **⚡ GPU Acceleration** - Optimized for NVIDIA GPUs
- **💰 Cost-Effective** - Pay only for actual processing time

## Serverless Functions

The handler supports three main actions:

### 1. `transcribe` - Audio Transcription
Process audio files with optional speaker diarization and word-level alignment.

### 2. `health` - Health Check
Check the status of the serverless function and model loading.

### 3. `models` - List Available Models
Get a list of available Whisper models.

## Quick Start

### Prerequisites

1. **RunPod Account** - Sign up at [https://www.runpod.io](https://www.runpod.io)
2. **GitHub Repository** - Fork or use this repository
3. **Hugging Face Token** - Optional, for speaker diarization

### Deployment

#### Step 1: Deploy to RunPod Serverless

1. Go to **[RunPod Dashboard](https://www.runpod.io/console)**
2. Navigate to **"Serverless"** → **"New Endpoint"**
3. Choose **"Custom Source"** → **"GitHub Repo"**
4. Configure:
   - **Repository**: `https://github.com/703deuce/whisper.git`
   - **Branch**: `master`
   - **Container Disk**: `20GB` minimum
   - **GPU**: Choose **RTX 3090**, **A10G**, or **T4** (8-12GB VRAM recommended)
   - **Timeout**: `300` seconds (5 minutes)
   - **Memory**: `16GB` recommended

#### Step 2: Environment Variables (Optional)

- `WHISPER_MODEL`: `large-v2` (default) or `large-v3`, `medium`, `base`
- `HF_TOKEN`: Your Hugging Face token (required for speaker diarization)

#### Step 3: Deploy

1. Click **"Create Endpoint"**
2. Wait for build completion (5-10 minutes)
3. Get your endpoint ID from the dashboard

## Usage Examples

### Using RunPod Python SDK

```python
import runpod
import base64

# Initialize RunPod client
runpod.api_key = "your-runpod-api-key"

# Read audio file
with open("audio.wav", "rb") as f:
    audio_content = f.read()

# Encode to base64
audio_base64 = base64.b64encode(audio_content).decode("utf-8")

# Basic transcription
response = runpod.run_sync(
    endpoint_id="your-endpoint-id",
    input={
        "action": "transcribe",
        "file_content": audio_base64,
        "filename": "audio.wav",
        "model": "whisper-large-v2"
    }
)

print(response)
```

### With Speaker Diarization

```python
response = runpod.run_sync(
    endpoint_id="your-endpoint-id",
    input={
        "action": "transcribe",
        "file_content": audio_base64,
        "filename": "audio.wav",
        "diarize": True,
        "align": True,
        "min_speakers": 2,
        "max_speakers": 4
    }
)

print(response)
```

### SRT Format Output

```python
response = runpod.run_sync(
    endpoint_id="your-endpoint-id",
    input={
        "action": "transcribe",
        "file_content": audio_base64,
        "filename": "audio.wav",
        "response_format": "srt"
    }
)

print(response["text"])  # SRT formatted subtitles
```

### Health Check

```python
response = runpod.run_sync(
    endpoint_id="your-endpoint-id",
    input={"action": "health"}
)

print(response)
```

### Using HTTP API

```bash
# Health check
curl -X POST "https://api.runpod.ai/v2/your-endpoint-id/run" \
  -H "Authorization: Bearer your-runpod-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "health"
    }
  }'

# Transcription (you need to base64 encode your audio file)
curl -X POST "https://api.runpod.ai/v2/your-endpoint-id/run" \
  -H "Authorization: Bearer your-runpod-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "transcribe",
      "file_content": "base64_encoded_audio_content",
      "filename": "audio.wav",
      "diarize": true,
      "align": true
    }
  }'
```

## Input Parameters

### Transcription Parameters

- `action`: `"transcribe"` (required)
- `file_content`: Base64-encoded audio file content (required)
- `filename`: Original filename with extension (required)
- `model`: Model to use (default: `"whisper-large-v2"`)
- `language`: Language code (auto-detect if not specified)
- `response_format`: Output format - `"json"`, `"srt"`, `"vtt"` (default: `"json"`)
- `timestamp_granularities`: `"segment"` or `"word"` (default: `"segment"`)
- `diarize`: Enable speaker diarization (default: `false`)
- `align`: Enable word-level alignment (default: `false`)
- `min_speakers`: Minimum number of speakers for diarization
- `max_speakers`: Maximum number of speakers for diarization

### Health Check Parameters

- `action`: `"health"` (required)

### Models List Parameters

- `action`: `"models"` (required)

## Response Format

### Transcription Response

```json
{
  "text": "Full transcription text...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.0,
      "text": "Segment text...",
      "speaker": "SPEAKER_00"
    }
  ],
  "language": "en",
  "word_timestamps": [...],
  "speakers": [...]
}
```

### Health Check Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "device": "cuda",
  "models_loaded": {
    "whisper": true,
    "align": false,
    "diarize": false
  }
}
```

## Performance & Optimization

### Model Selection
- **`large-v2`**: Best accuracy (default)
- **`large-v3`**: Latest model, improved performance
- **`medium`**: Good balance of speed and accuracy
- **`base`**: Fastest inference

### GPU Recommendations
- **RTX 3090**: Best performance for large models
- **A10G**: Good balance of performance and cost
- **T4**: Budget option, suitable for smaller models

### Cost Optimization
- **Auto-scaling**: Function scales to zero when not in use
- **Pay-per-use**: Only charged for actual processing time
- **Batch processing**: Process multiple files in sequence

## Supported Audio Formats

- WAV, MP3, MP4, M4A, FLAC, OGG
- Sample rates: 8kHz - 48kHz
- Channels: Mono or Stereo
- Maximum file size: Depends on your RunPod configuration

## Troubleshooting

### Common Issues

1. **Build failures**: Check Dockerfile and requirements.txt
2. **Memory errors**: Use smaller batch size or model
3. **Timeout errors**: Increase timeout setting in RunPod
4. **Diarization not working**: Ensure HF_TOKEN is set correctly

### Error Responses

```json
{
  "error": "Error message describing the issue"
}
```

## Local Development

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- 8GB+ GPU memory

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_hugging_face_token
export WHISPER_MODEL=large-v2

# Test the handler locally
python handler.py
```

## License

This project is open source and available under the MIT License. 