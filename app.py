import os
import tempfile
import uuid
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import whisperx
import torch
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhisperX API Server",
    description="OpenAI Whisper API compatible server with speaker diarization and word alignment",
    version="1.0.0"
)

# Global variables for model caching
whisper_model = None
align_model = None
diarize_model = None
device = None
compute_type = None

class TranscriptionResponse(BaseModel):
    text: str
    segments: Optional[List[dict]] = None
    language: Optional[str] = None
    word_timestamps: Optional[List[dict]] = None
    speakers: Optional[List[dict]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    device: str
    models_loaded: dict

class ModelInfo(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

def initialize_models():
    """Initialize WhisperX models"""
    global whisper_model, align_model, diarize_model, device, compute_type
    
    # Determine device and compute type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    
    logger.info(f"Using device: {device}, compute_type: {compute_type}")
    
    # Load whisper model
    model_name = os.getenv("WHISPER_MODEL", "large-v2")
    try:
        whisper_model = whisperx.load_model(model_name, device, compute_type=compute_type)
        logger.info(f"Loaded Whisper model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

def load_align_model(language_code: str):
    """Load alignment model for specific language"""
    global align_model
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        return align_model, metadata
    except Exception as e:
        logger.error(f"Failed to load alignment model for {language_code}: {e}")
        return None, None

def load_diarize_model():
    """Load diarization model"""
    global diarize_model
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not found, diarization may not work")
            return None
        
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        logger.info("Loaded diarization model")
        return diarize_model
    except Exception as e:
        logger.error(f"Failed to load diarization model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()

@app.get("/healthcheck", response_model=HealthResponse)
async def healthcheck():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        device=device or "unknown",
        models_loaded={
            "whisper": whisper_model is not None,
            "align": align_model is not None,
            "diarize": diarize_model is not None
        }
    )

@app.get("/models/list")
async def list_models():
    """List available models"""
    models = [
        ModelInfo(
            id="whisper-large-v2",
            object="model",
            created=1677610602,
            owned_by="openai"
        ),
        ModelInfo(
            id="whisper-large-v3",
            object="model", 
            created=1677610602,
            owned_by="openai"
        ),
        ModelInfo(
            id="whisper-medium",
            object="model",
            created=1677610602,
            owned_by="openai"
        ),
        ModelInfo(
            id="whisper-small",
            object="model",
            created=1677610602,
            owned_by="openai"
        )
    ]
    return {"data": models}

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v2"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: str = Form("segment"),
    diarize: bool = Form(False),
    align: bool = Form(False),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    """Transcribe audio with optional diarization and alignment"""
    
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Processing file: {file.filename} (ID: {file_id})")
        
        # Load audio and transcribe
        audio = whisperx.load_audio(tmp_file_path)
        result = whisper_model.transcribe(audio, batch_size=16, language=language)
        
        response_data = {
            "text": result["segments"][0]["text"] if result["segments"] else "",
            "segments": result["segments"],
            "language": result.get("language", "en")
        }
        
        # Word-level alignment
        if align and result["segments"]:
            language_code = result["language"]
            align_model_inst, metadata = load_align_model(language_code)
            if align_model_inst:
                result = whisperx.align(result["segments"], align_model_inst, metadata, audio, device, return_char_alignments=False)
                response_data["segments"] = result["segments"]
                
                # Extract word timestamps if requested
                if "word" in timestamp_granularities:
                    word_timestamps = []
                    for segment in result["segments"]:
                        if "words" in segment:
                            for word in segment["words"]:
                                word_timestamps.append({
                                    "word": word["word"],
                                    "start": word["start"],
                                    "end": word["end"]
                                })
                    response_data["word_timestamps"] = word_timestamps
        
        # Speaker diarization
        if diarize:
            diarize_model_inst = load_diarize_model()
            if diarize_model_inst:
                diarize_segments = diarize_model_inst(audio, min_speakers=min_speakers, max_speakers=max_speakers)
                result = whisperx.assign_speaker_labels(result, diarize_segments)
                response_data["segments"] = result["segments"]
                
                # Extract speaker information
                speakers = []
                for segment in result["segments"]:
                    if "speaker" in segment:
                        speakers.append({
                            "speaker": segment["speaker"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"]
                        })
                response_data["speakers"] = speakers
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Format response based on response_format
        if response_format == "text":
            return response_data["text"]
        elif response_format == "srt":
            # Convert to SRT format
            srt_content = ""
            for i, segment in enumerate(response_data["segments"], 1):
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                srt_content += f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n"
            return srt_content
        elif response_format == "vtt":
            # Convert to VTT format
            vtt_content = "WEBVTT\n\n"
            for segment in response_data["segments"]:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                vtt_content += f"{start_time} --> {end_time}\n{segment['text']}\n\n"
            return vtt_content
        else:
            return response_data
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # Clean up temporary file on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT/VTT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "WhisperX API Server is running", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 