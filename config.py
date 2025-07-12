import os
from typing import Optional, List

class Config:
    """Configuration settings for WhisperX API Server"""
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Whisper Model Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v2")
    PRELOAD_MODELS: bool = os.getenv("PRELOAD_MODELS", "true").lower() == "true"
    CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "5"))
    
    # Hugging Face Configuration
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    
    # CUDA Configuration
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "float16")
    
    # Performance Settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Audio Processing Settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    SUPPORTED_FORMATS: List[str] = os.getenv("SUPPORTED_FORMATS", "wav,mp3,mp4,flac,ogg,m4a,webm").split(",")
    
    # Diarization Settings
    DEFAULT_MIN_SPEAKERS: int = int(os.getenv("DEFAULT_MIN_SPEAKERS", "1"))
    DEFAULT_MAX_SPEAKERS: int = int(os.getenv("DEFAULT_MAX_SPEAKERS", "10"))
    DIARIZATION_TIMEOUT: int = int(os.getenv("DIARIZATION_TIMEOUT", "300"))
    
    # Alignment Settings
    ENABLE_ALIGNMENT: bool = os.getenv("ENABLE_ALIGNMENT", "true").lower() == "true"
    ALIGNMENT_TIMEOUT: int = int(os.getenv("ALIGNMENT_TIMEOUT", "60"))
    
    # RunPod Specific Settings
    RUNPOD_ENDPOINT_ID: Optional[str] = os.getenv("RUNPOD_ENDPOINT_ID")
    RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
    
    # File paths
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/whisper")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/root/.cache")
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported Whisper models"""
        return [
            "whisper-small",
            "whisper-medium", 
            "whisper-large-v2",
            "whisper-large-v3"
        ]
    
    @classmethod
    def validate_model(cls, model_name: str) -> bool:
        """Validate if model name is supported"""
        return model_name in cls.get_supported_models()
    
    @classmethod
    def get_model_config(cls, model_name: str) -> dict:
        """Get configuration for specific model"""
        model_configs = {
            "whisper-small": {
                "name": "small",
                "vram_requirement": "2GB",
                "recommended_batch_size": 32
            },
            "whisper-medium": {
                "name": "medium", 
                "vram_requirement": "4GB",
                "recommended_batch_size": 24
            },
            "whisper-large-v2": {
                "name": "large-v2",
                "vram_requirement": "8GB",
                "recommended_batch_size": 16
            },
            "whisper-large-v3": {
                "name": "large-v3",
                "vram_requirement": "10GB",
                "recommended_batch_size": 12
            }
        }
        return model_configs.get(model_name, {})
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
    
    @classmethod
    def log_config(cls):
        """Log current configuration (excluding sensitive data)"""
        config_info = {
            "host": cls.HOST,
            "port": cls.PORT,
            "whisper_model": cls.WHISPER_MODEL,
            "compute_type": cls.COMPUTE_TYPE,
            "batch_size": cls.BATCH_SIZE,
            "log_level": cls.LOG_LEVEL,
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "supported_formats": cls.SUPPORTED_FORMATS,
            "enable_alignment": cls.ENABLE_ALIGNMENT,
            "hf_token_set": cls.HF_TOKEN is not None,
            "cuda_device": cls.CUDA_VISIBLE_DEVICES
        }
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration loaded: {config_info}")
        
        return config_info

# Global configuration instance
config = Config() 