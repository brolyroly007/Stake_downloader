"""
Configuración centralizada del proyecto usando Pydantic Settings.
Carga variables desde .env y proporciona validación de tipos.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración principal del proyecto Stake Viral Automation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===================
    # GENERAL
    # ===================
    debug: bool = Field(default=True, description="Modo debug")
    log_level: str = Field(default="INFO", description="Nivel de logging")

    # ===================
    # KICK CONFIGURATION
    # ===================
    kick_channels: str = Field(
        default="trainwreckstv,roshtein,xposed",
        description="Canales de Kick a monitorear (separados por coma)",
    )
    kick_websocket_url: str = Field(
        default="wss://ws-us2.pusher.com/app/eb1d5f283081a78b932c",
        description="URL del WebSocket de Kick",
    )

    @property
    def kick_channels_list(self) -> List[str]:
        """Retorna lista de canales de Kick."""
        return [ch.strip() for ch in self.kick_channels.split(",") if ch.strip()]

    # ===================
    # STAKE CONFIGURATION
    # ===================
    stake_api_url: str = Field(
        default="https://stake.com/_api/graphql",
        description="URL de la API GraphQL de Stake",
    )
    stake_ws_url: str = Field(
        default="wss://stake.com/_api/websocket",
        description="URL del WebSocket de Stake",
    )
    big_win_threshold: float = Field(
        default=10000.0,
        description="Umbral en USD para considerar un Big Win",
    )

    # ===================
    # DETECTION THRESHOLDS
    # ===================
    chat_velocity_threshold: int = Field(
        default=50,
        description="Mensajes por segundo para considerar viral",
    )
    audio_peak_threshold: float = Field(
        default=-20.0,
        description="Nivel de audio (dB) para detectar gritos",
    )
    virality_score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Puntuación mínima de viralidad (0-1)",
    )

    # ===================
    # VIDEO PROCESSING
    # ===================
    output_dir: Path = Field(
        default=Path("./output"),
        description="Directorio de salida para videos procesados",
    )
    clips_dir: Path = Field(
        default=Path("./clips"),
        description="Directorio para clips descargados",
    )
    temp_dir: Path = Field(
        default=Path("./temp"),
        description="Directorio temporal",
    )
    video_format: str = Field(default="mp4", description="Formato de video")
    video_codec: str = Field(default="libx264", description="Codec de video")
    audio_codec: str = Field(default="aac", description="Codec de audio")
    vertical_width: int = Field(default=1080, description="Ancho para formato vertical")
    vertical_height: int = Field(default=1920, description="Alto para formato vertical")

    # ===================
    # PROXIES
    # ===================
    use_proxies: bool = Field(default=False, description="Usar proxies rotativos")
    proxy_list_file: Optional[Path] = Field(
        default=None,
        description="Archivo con lista de proxies",
    )
    proxy_rotation_interval: int = Field(
        default=60,
        description="Intervalo de rotación de proxies (segundos)",
    )

    # ===================
    # WHISPER
    # ===================
    whisper_model: str = Field(
        default="base",
        description="Modelo de Whisper (tiny, base, small, medium, large)",
    )

    # ===================
    # SECURITY
    # ===================
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        description="User Agent para requests HTTP",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Valida que el nivel de log sea válido."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level debe ser uno de: {valid_levels}")
        return v_upper

    @field_validator("whisper_model")
    @classmethod
    def validate_whisper_model(cls, v: str) -> str:
        """Valida que el modelo de Whisper sea válido."""
        valid_models = ["tiny", "base", "small", "medium", "large"]
        v_lower = v.lower()
        if v_lower not in valid_models:
            raise ValueError(f"whisper_model debe ser uno de: {valid_models}")
        return v_lower

    def ensure_directories(self) -> None:
        """Crea los directorios necesarios si no existen."""
        for dir_path in [self.output_dir, self.clips_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_proxy_list(self) -> List[str]:
        """Carga la lista de proxies desde el archivo."""
        if not self.use_proxies or not self.proxy_list_file:
            return []

        if not self.proxy_list_file.exists():
            return []

        with open(self.proxy_list_file, "r") as f:
            return [line.strip() for line in f if line.strip()]


# Instancia global de configuración
settings = Settings()
