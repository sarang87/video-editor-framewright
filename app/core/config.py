from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Application Config
    APP_NAME: str = "Video Editor Framewright"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API Keys & External Services
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # vLLM Configuration
    VLLM_BASE_URL: str = "http://vllm:8000/v1"
    # Matches the default in docker-compose/start_server but can be overridden
    MODEL_NAME: str = "Qwen/Qwen3-VL-8B-Instruct-FP8" 
    MAX_MODEL_LEN: int = 8192
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()
