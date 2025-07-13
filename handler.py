#!/usr/bin/env python3
"""
RunPod Serverless Handler for WhisperX API
Supports transcription, health check, and model listing
"""

import os
import tempfile
import uuid
import base64
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import runpod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
whisper_model = None
align_model = None
diarize_model = None
device = None
compute_type = None
models_initialized = False

def initialize_models():
    """Initialize WhisperX models with better error handling"""
    global whisper_model, align_model, diarize_model, device, compute_type, models_initialized
    
    try:
        # Import here to avoid import errors during container build
        import whisperx
        import torch
        
        logger.info("Initializing WhisperX models...")
        
        # Set device and compute type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Using device: {device}, compute_type: {compute_type}")
        
        # Load main WhisperX model
        model_name = os.getenv("WHISPER_MODEL", "large-v2")
        logger.info(f"Loading WhisperX model: {model_name}")
        
        whisper_model = whisperx.load_model(
            model_name, 
            device=device, 
            compute_type=compute_type
        )
        
        models_initialized = True
        logger.info("✅ WhisperX models initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {str(e)}")
        models_initialized = False
        return False

def load_align_model(language_code: str):
    """Load alignment model for specific language"""
    global align_model
    try:
        import whisperx
        if not align_model:
            align_model, _ = whisperx.load_align_model(language_code=language_code, device=device)
        return align_model
    except Exception as e:
        logger.error(f"Failed to load alignment model: {str(e)}")
        return None

def load_diarize_model():
    """Load diarization model"""
    global diarize_model
    try:
        import whisperx
        if not diarize_model:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable required for diarization")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        return diarize_model
    except Exception as e:
        logger.error(f"Failed to load diarization model: {str(e)}")
        return None

def process_audio_file(file_content: bytes, filename: str) -> str:
    """Save audio file content to temporary file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as tmp_file:
        tmp_file.write(file_content)
        return tmp_file.name

def transcribe_audio(
    file_content: bytes,
    filename: str,
    model: str = "whisper-large-v2",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "json",
    temperature: float = 0.0,
    timestamp_granularities: List[str] = ["segment"],
    stream: bool = False,
    hotwords: Optional[str] = None,
    suppress_numerals: bool = True,
    highlight_words: bool = False,
    diarize: bool = False,
    align: bool = True,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> Dict[str, Any]:
    """Process audio transcription request"""
    
    if not models_initialized:
        if not initialize_models():
            return {"error": "Models not initialized"}
    
    try:
        # Save uploaded file
        audio_path = process_audio_file(file_content, filename)
        
        try:
            # Load audio and transcribe
            import whisperx
            audio = whisperx.load_audio(audio_path)
            result = whisper_model.transcribe(audio, batch_size=16)
            
            # Get detected language
            detected_language = result.get("language", "en")
            if language:
                detected_language = language
            
            # Word-level alignment if requested
            if align and result["segments"]:
                align_model_loaded = load_align_model(detected_language)
                if align_model_loaded:
                    result = whisperx.align(result["segments"], align_model_loaded, whisperx.utils.load_align_model(detected_language, device)[1], audio, device, return_char_alignments=False)
            
            # Speaker diarization if requested
            speakers_info = []
            if diarize:
                diarize_model_loaded = load_diarize_model()
                if diarize_model_loaded:
                    diarize_segments = diarize_model_loaded(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    # Extract speaker information
                    for segment in result.get("segments", []):
                        if "speaker" in segment:
                            speaker_info = {
                                "speaker": segment["speaker"],
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": segment["text"]
                            }
                            speakers_info.append(speaker_info)
            
            # Format response based on requested format
            if response_format == "srt":
                return {"text": format_srt(result["segments"]), "format": "srt"}
            elif response_format == "vtt":
                return {"text": format_vtt(result["segments"]), "format": "vtt"}
            else:
                # JSON response
                response = {
                    "text": result["text"],
                    "segments": result["segments"],
                    "language": detected_language,
                    "word_timestamps": [] if timestamp_granularities == "word" else None,
                    "speakers": speakers_info if speakers_info else None
                }
                
                # Add word-level timestamps if requested
                if "word" in timestamp_granularities:
                    for segment in result.get("segments", []):
                        if "words" in segment:
                            response["word_timestamps"].extend(segment["words"])
                
                return response
                
        finally:
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return {"error": f"Transcription failed: {str(e)}"}

def translate_audio(
    file_content: bytes,
    filename: str,
    model: str = "whisper-large-v2",
    prompt: Optional[str] = None,
    response_format: str = "json",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Process audio translation request (translate to English)"""
    
    if not models_initialized:
        if not initialize_models():
            return {"error": "Models not initialized"}
    
    try:
        # Save uploaded file
        audio_path = process_audio_file(file_content, filename)
        
        try:
            # Load audio and transcribe (WhisperX translates to English automatically)
            import whisperx
            audio = whisperx.load_audio(audio_path)
            
            # For translation, we force the task to be 'translate' and output language to English
            result = whisper_model.transcribe(
                audio, 
                batch_size=16, 
                task="translate",  # This forces translation to English
                language=None      # Auto-detect source language
            )
            
            # Format response based on requested format
            if response_format == "srt":
                return {"text": format_srt(result["segments"]), "format": "srt"}
            elif response_format == "vtt":
                return {"text": format_vtt(result["segments"]), "format": "vtt"}
            else:
                # JSON response
                return {
                    "text": result["text"],
                    "segments": result["segments"],
                    "language": "en"  # Always English for translations
                }
                
        finally:
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return {"error": f"Translation failed: {str(e)}"}

def load_model(model_name: str) -> Dict[str, Any]:
    """Load a specific model into memory"""
    global whisper_model, models_initialized
    
    try:
        import whisperx
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Loading model: {model_name}")
        
        # Load the requested model
        whisper_model = whisperx.load_model(
            model_name, 
            device=device, 
            compute_type=compute_type
        )
        
        models_initialized = True
        logger.info(f"✅ Model {model_name} loaded successfully!")
        
        return {
            "status": "success",
            "message": f"Model {model_name} loaded successfully",
            "model": model_name,
            "device": device,
            "compute_type": compute_type
        }
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return {"error": f"Failed to load model {model_name}: {str(e)}"}

def unload_model(model_name: str) -> Dict[str, Any]:
    """Unload a specific model from memory"""
    global whisper_model, align_model, diarize_model, models_initialized
    
    try:
        if model_name == "whisper" or model_name == "all":
            whisper_model = None
            logger.info("Whisper model unloaded")
        
        if model_name == "align" or model_name == "all":
            align_model = None
            logger.info("Align model unloaded")
        
        if model_name == "diarize" or model_name == "all":
            diarize_model = None
            logger.info("Diarize model unloaded")
        
        if model_name == "all":
            models_initialized = False
            logger.info("All models unloaded")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return {
            "status": "success",
            "message": f"Model {model_name} unloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {str(e)}")
        return {"error": f"Failed to unload model {model_name}: {str(e)}"}

def format_srt(segments: List[Dict]) -> str:
    """Format segments as SRT subtitle file"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        
        srt_content += f"{i}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{text}\n\n"
    
    return srt_content.strip()

def format_vtt(segments: List[Dict]) -> str:
    """Format segments as VTT subtitle file"""
    vtt_content = "WEBVTT\n\n"
    for segment in segments:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        
        vtt_content += f"{start_time} --> {end_time}\n"
        vtt_content += f"{text}\n\n"
    
    return vtt_content.strip()

def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy" if models_initialized else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": device or "unknown",
        "models_loaded": {
            "whisper": whisper_model is not None,
            "align": align_model is not None,
            "diarize": diarize_model is not None
        }
    }

def list_models() -> Dict[str, Any]:
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai-internal"
            },
            {
                "id": "whisper-large-v2",
                "object": "model", 
                "created": 1677610602,
                "owned_by": "openai-internal"
            },
            {
                "id": "whisper-large-v3",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai-internal"
            }
        ]
    }

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod Serverless handler function
    
    Expected input format:
    {
        "input": {
            "action": "transcribe" | "translate" | "health" | "models" | "load_model" | "unload_model",
            "file_content": "base64_encoded_audio_file",  # for transcribe/translate
            "filename": "audio.wav",                        # for transcribe/translate
            "model": "whisper-large-v2",                   # optional
            "language": "en",                              # optional (transcribe only)
            "prompt": "optional prompt",                   # optional
            "response_format": "json",                     # optional
            "temperature": 0.0,                            # optional
            "timestamp_granularities": ["segment"],        # optional list
            "stream": false,                               # optional
            "hotwords": "optional hotwords",               # optional
            "suppress_numerals": true,                     # optional
            "highlight_words": false,                      # optional
            "diarize": false,                              # optional
            "align": true,                                 # optional
            "min_speakers": 2,                             # optional
            "max_speakers": 4,                             # optional
            "model_name": "whisper-large-v2"               # for load_model/unload_model
        }
    }
    """
    
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "transcribe")
        
        logger.info(f"Processing action: {action}")
        
        if action == "health":
            return health_check()
        
        elif action == "models":
            return list_models()
        
        elif action == "load_model":
            model_name = input_data.get("model_name", "large-v2")
            return load_model(model_name)
        
        elif action == "unload_model":
            model_name = input_data.get("model_name", "whisper")
            return unload_model(model_name)
        
        elif action == "translate":
            # Validate required parameters
            if "file_content" not in input_data:
                return {"error": "file_content is required for translation"}
            
            if "filename" not in input_data:
                return {"error": "filename is required for translation"}
            
            # Decode base64 file content
            try:
                file_content = base64.b64decode(input_data["file_content"])
            except Exception as e:
                return {"error": f"Invalid base64 file content: {str(e)}"}
            
            # Extract parameters
            filename = input_data.get("filename", "audio.wav")
            model = input_data.get("model", "whisper-large-v2")
            prompt = input_data.get("prompt")
            response_format = input_data.get("response_format", "json")
            temperature = input_data.get("temperature", 0.0)
            
            return translate_audio(
                file_content=file_content,
                filename=filename,
                model=model,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature
            )
        
        elif action == "transcribe":
            # Validate required parameters
            if "file_content" not in input_data:
                return {"error": "file_content is required for transcription"}
            
            if "filename" not in input_data:
                return {"error": "filename is required for transcription"}
            
            # Decode base64 file content
            try:
                file_content = base64.b64decode(input_data["file_content"])
            except Exception as e:
                return {"error": f"Invalid base64 file content: {str(e)}"}
            
            # Extract parameters
            filename = input_data.get("filename", "audio.wav")
            model = input_data.get("model", "whisper-large-v2")
            language = input_data.get("language")
            prompt = input_data.get("prompt")
            response_format = input_data.get("response_format", "json")
            temperature = input_data.get("temperature", 0.0)
            timestamp_granularities = input_data.get("timestamp_granularities", ["segment"])
            stream = input_data.get("stream", False)
            hotwords = input_data.get("hotwords")
            suppress_numerals = input_data.get("suppress_numerals", True)
            highlight_words = input_data.get("highlight_words", False)
            diarize = input_data.get("diarize", False)
            align = input_data.get("align", True)
            min_speakers = input_data.get("min_speakers")
            max_speakers = input_data.get("max_speakers")
            
            return transcribe_audio(
                file_content=file_content,
                filename=filename,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                stream=stream,
                hotwords=hotwords,
                suppress_numerals=suppress_numerals,
                highlight_words=highlight_words,
                diarize=diarize,
                align=align,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": f"Handler failed: {str(e)}"}

# Initialize models on startup
initialize_models()

# RunPod serverless handler
runpod.serverless.start({"handler": handler}) 