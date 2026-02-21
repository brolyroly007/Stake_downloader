"""
Sistema de logging centralizado usando Loguru.
Proporciona logging estructurado, rotación de archivos y colores en consola.
"""

import sys
from pathlib import Path
from loguru import logger as _logger

from .config import settings


def setup_logger():
    """Configura el logger con las opciones del proyecto."""

    # Remover handler por defecto
    _logger.remove()

    # Formato para consola (con colores)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Formato para archivo (sin colores)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # Handler para consola
    _logger.add(
        sys.stderr,
        format=console_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )

    # Crear directorio de logs
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Handler para archivo principal
    _logger.add(
        logs_dir / "app_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="DEBUG" if settings.debug else "INFO",
        rotation="100 MB",  # Rotar cuando llegue a 100 MB
        retention="30 days",  # Mantener logs por 30 días
        compression="zip",  # Comprimir logs antiguos
        encoding="utf-8",
    )

    # Handler para errores (archivo separado)
    _logger.add(
        logs_dir / "errors_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="ERROR",
        rotation="50 MB",
        retention="60 days",
        compression="zip",
        encoding="utf-8",
    )

    # Handler para eventos de viralidad (archivo separado para análisis)
    _logger.add(
        logs_dir / "viral_events_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="INFO",
        filter=lambda record: "viral" in record["extra"].get("tags", []),
        rotation="50 MB",
        retention="90 days",
        encoding="utf-8",
    )

    _logger.info("Logger inicializado correctamente")
    _logger.debug(f"Nivel de log: {settings.log_level}")
    _logger.debug(f"Modo debug: {settings.debug}")

    return _logger


# Funciones helper para logging estructurado
def log_viral_event(event_type: str, score: float, details: dict):
    """Log específico para eventos virales detectados."""
    _logger.bind(tags=["viral"]).info(
        f"VIRAL EVENT | Type: {event_type} | Score: {score:.2f} | Details: {details}"
    )


def log_clip_downloaded(source: str, clip_id: str, path: str):
    """Log cuando se descarga un clip."""
    _logger.info(f"CLIP DOWNLOADED | Source: {source} | ID: {clip_id} | Path: {path}")


def log_video_processed(input_path: str, output_path: str, duration: float):
    """Log cuando se procesa un video."""
    _logger.info(
        f"VIDEO PROCESSED | Input: {input_path} | Output: {output_path} | Duration: {duration:.2f}s"
    )


def log_upload_success(platform: str, video_id: str, url: str):
    """Log cuando se sube un video exitosamente."""
    _logger.success(f"UPLOAD SUCCESS | Platform: {platform} | ID: {video_id} | URL: {url}")


def log_upload_failed(platform: str, error: str):
    """Log cuando falla una subida."""
    _logger.error(f"UPLOAD FAILED | Platform: {platform} | Error: {error}")


def log_monitor_connected(platform: str, channel: str):
    """Log cuando se conecta un monitor."""
    _logger.info(f"MONITOR CONNECTED | Platform: {platform} | Channel: {channel}")


def log_monitor_disconnected(platform: str, channel: str, reason: str = ""):
    """Log cuando se desconecta un monitor."""
    _logger.warning(
        f"MONITOR DISCONNECTED | Platform: {platform} | Channel: {channel} | Reason: {reason}"
    )


# Inicializar logger al importar el módulo
logger = setup_logger()
