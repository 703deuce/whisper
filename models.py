import os
import torch
import whisperx
import logging
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and caching of WhisperX models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        
        # Model caches
        self.whisper_models: Dict[str, Any] = {}
        self.align_models: Dict[str, Tuple[Any, Any]] = {}
        self.diarize_model: Optional[Any] = None
        
        logger.info(f"ModelManager initialized with device: {self.device}, compute_type: {self.compute_type}")
    
    def get_whisper_model(self, model_name: str = "large-v2") -> Any:
        """Get or load a Whisper model"""
        if model_name not in self.whisper_models:
            logger.info(f"Loading Whisper model: {model_name}")
            try:
                model = whisperx.load_model(
                    model_name, 
                    device=self.device, 
                    compute_type=self.compute_type,
                    language="en"  # Default language
                )
                self.whisper_models[model_name] = model
                logger.info(f"Successfully loaded Whisper model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model {model_name}: {e}")
                raise
        
        return self.whisper_models[model_name]
    
    def get_align_model(self, language_code: str) -> Tuple[Any, Any]:
        """Get or load an alignment model for a specific language"""
        if language_code not in self.align_models:
            logger.info(f"Loading alignment model for language: {language_code}")
            try:
                model, metadata = whisperx.load_align_model(
                    language_code=language_code, 
                    device=self.device
                )
                self.align_models[language_code] = (model, metadata)
                logger.info(f"Successfully loaded alignment model for: {language_code}")
            except Exception as e:
                logger.error(f"Failed to load alignment model for {language_code}: {e}")
                return None, None
        
        return self.align_models[language_code]
    
    def get_diarize_model(self) -> Optional[Any]:
        """Get or load the diarization model"""
        if self.diarize_model is None:
            logger.info("Loading diarization model")
            try:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    logger.warning("HF_TOKEN environment variable not set. Diarization will not be available.")
                    return None
                
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=self.device
                )
                logger.info("Successfully loaded diarization model")
            except Exception as e:
                logger.error(f"Failed to load diarization model: {e}")
                return None
        
        return self.diarize_model
    
    def clear_cache(self):
        """Clear all model caches"""
        logger.info("Clearing model caches")
        self.whisper_models.clear()
        self.align_models.clear()
        self.diarize_model = None
        
        # Force garbage collection
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "device": self.device,
            "compute_type": self.compute_type,
            "whisper_models_loaded": list(self.whisper_models.keys()),
            "align_models_loaded": list(self.align_models.keys()),
            "diarize_model_loaded": self.diarize_model is not None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

# Global model manager instance
model_manager = ModelManager()

# Supported models configuration
SUPPORTED_MODELS = {
    "whisper-large-v2": {
        "name": "large-v2",
        "description": "Whisper large-v2 model",
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "hu", "cs", "sk", "sl", "hr", "bg", "ro", "uk", "el", "he", "fa", "ur", "th", "vi", "id", "ms", "tl", "sw", "zu", "af", "sq", "am", "hy", "az", "eu", "be", "bn", "bs", "ca", "ceb", "co", "cy", "eo", "et", "fi", "fy", "ga", "gd", "gl", "gu", "ha", "haw", "is", "ig", "ja", "jw", "ka", "kk", "km", "kn", "ku", "ky", "lo", "la", "lv", "lt", "lb", "mk", "mg", "ml", "mn", "mr", "mt", "my", "ne", "ny", "ps", "pa", "si", "sk", "so", "su", "sv", "sw", "ta", "te", "tg", "ti", "to", "tr", "uk", "ur", "uz", "vi", "xh", "yi", "yo", "zu"]
    },
    "whisper-large-v3": {
        "name": "large-v3",
        "description": "Whisper large-v3 model (latest)",
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "hu", "cs", "sk", "sl", "hr", "bg", "ro", "uk", "el", "he", "fa", "ur", "th", "vi", "id", "ms", "tl", "sw", "zu", "af", "sq", "am", "hy", "az", "eu", "be", "bn", "bs", "ca", "ceb", "co", "cy", "eo", "et", "fi", "fy", "ga", "gd", "gl", "gu", "ha", "haw", "is", "ig", "ja", "jw", "ka", "kk", "km", "kn", "ku", "ky", "lo", "la", "lv", "lt", "lb", "mk", "mg", "ml", "mn", "mr", "mt", "my", "ne", "ny", "ps", "pa", "si", "sk", "so", "su", "sv", "sw", "ta", "te", "tg", "ti", "to", "tr", "uk", "ur", "uz", "vi", "xh", "yi", "yo", "zu"]
    },
    "whisper-medium": {
        "name": "medium",
        "description": "Whisper medium model",
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "hu", "cs", "sk", "sl", "hr", "bg", "ro", "uk", "el", "he", "fa", "ur", "th", "vi", "id", "ms", "tl", "sw", "zu"]
    },
    "whisper-small": {
        "name": "small",
        "description": "Whisper small model",
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "hu", "cs", "sk", "sl", "hr", "bg", "ro", "uk", "el", "he", "fa", "ur", "th", "vi", "id", "ms", "tl", "sw", "zu"]
    }
}

def get_supported_models() -> Dict[str, Dict[str, Any]]:
    """Get dictionary of supported models"""
    return SUPPORTED_MODELS

def validate_model_name(model_name: str) -> bool:
    """Validate if model name is supported"""
    return model_name in SUPPORTED_MODELS

def get_model_languages(model_name: str) -> list:
    """Get supported languages for a model"""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]["languages"]
    return [] 