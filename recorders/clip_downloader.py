"""
Descargador de clips usando yt-dlp.

Soporta múltiples plataformas:
- Kick
- Twitch
- YouTube
- TikTok
- Y más...
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import json

# yt-dlp se ejecuta como proceso externo para mejor compatibilidad
# También se puede usar como librería: import yt_dlp


class ClipSource(Enum):
    """Plataformas soportadas para descarga."""
    KICK = "kick"
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    UNKNOWN = "unknown"


@dataclass
class ClipInfo:
    """Información de un clip descargado."""
    clip_id: str
    source: ClipSource
    url: str
    title: str
    duration: float  # segundos
    file_path: Path
    thumbnail_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    downloaded_at: datetime = field(default_factory=datetime.now)

    @property
    def filename(self) -> str:
        return self.file_path.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "source": self.source.value,
            "url": self.url,
            "title": self.title,
            "duration": self.duration,
            "file_path": str(self.file_path),
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "downloaded_at": self.downloaded_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DownloadOptions:
    """Opciones de descarga."""
    format: str = "best"  # best, bestvideo+bestaudio, worst, etc.
    max_resolution: Optional[int] = 1080  # 720, 1080, 1440, 2160
    download_thumbnail: bool = True
    download_subtitles: bool = False
    subtitle_languages: List[str] = field(default_factory=lambda: ["en", "es"])
    cookies_file: Optional[Path] = None  # Para sitios con login
    proxy: Optional[str] = None
    rate_limit: Optional[str] = None  # ej: "1M" para 1MB/s
    retries: int = 3
    timeout: int = 60


class ClipDownloader:
    """
    Descargador de clips multiplataforma usando yt-dlp.

    Ejemplo de uso:
        downloader = ClipDownloader(output_dir=Path("./clips"))
        clip = await downloader.download("https://kick.com/roshtein/clips/...")
        print(f"Descargado: {clip.file_path}")
    """

    def __init__(
        self,
        output_dir: Path,
        temp_dir: Optional[Path] = None,
        default_options: Optional[DownloadOptions] = None,
    ):
        """
        Args:
            output_dir: Directorio para guardar clips
            temp_dir: Directorio temporal (opcional)
            default_options: Opciones por defecto
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else self.output_dir / "temp"
        self.options = default_options or DownloadOptions()

        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Estadísticas
        self._stats = {
            "total_downloads": 0,
            "successful": 0,
            "failed": 0,
            "total_bytes": 0,
        }
        
        # Cache simple en memoria para info de videos
        self._info_cache: Dict[str, Dict[str, Any]] = {}

    def _detect_source(self, url: str) -> ClipSource:
        """Detecta la plataforma basándose en la URL."""
        url_lower = url.lower()

        if "kick.com" in url_lower:
            return ClipSource.KICK
        elif "twitch.tv" in url_lower or "clips.twitch.tv" in url_lower:
            return ClipSource.TWITCH
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            return ClipSource.YOUTUBE
        elif "tiktok.com" in url_lower:
            return ClipSource.TIKTOK
        elif "twitter.com" in url_lower or "x.com" in url_lower:
            return ClipSource.TWITTER
        elif "instagram.com" in url_lower:
            return ClipSource.INSTAGRAM
        else:
            return ClipSource.UNKNOWN

    def _build_ytdlp_args(
        self,
        url: str,
        output_template: str,
        options: DownloadOptions,
    ) -> List[str]:
        """Construye los argumentos para yt-dlp."""
        # Buscar yt-dlp en el venv primero
        venv_ytdlp = Path(__file__).parent.parent / ".venv" / "Scripts" / "yt-dlp.exe"
        ytdlp_cmd = str(venv_ytdlp) if venv_ytdlp.exists() else "yt-dlp"

        args = [
            ytdlp_cmd,
            "--no-playlist",
            "-o", output_template,
            "--write-info-json",
        ]

        # FFmpeg location (winget instala en esta ruta)
        ffmpeg_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        for p in ffmpeg_path.glob("Gyan.FFmpeg*/ffmpeg-*/bin"):
            if (p / "ffmpeg.exe").exists():
                args.extend(["--ffmpeg-location", str(p)])
                break

        # Optimización: Aria2c (descargas más rápidas)
        # Se asume que aria2c está en el PATH o en una ubicación conocida
        args.extend(["--external-downloader", "aria2c"])
        args.extend(["--external-downloader-args", "-x 8 -s 8 -k 1M"])

        # Formato optimizado: Preferir mp4 para evitar merge si es posible
        if options.max_resolution:
            # Intentar obtener mp4 directo primero, si no bestvideo+bestaudio
            res = options.max_resolution
            args.extend([
                "-f", 
                f"bestvideo[height<={res}][ext=mp4]+bestaudio[ext=m4a]/best[height<={res}][ext=mp4]/best[height<={res}]"
            ])
        else:
            args.extend(["-f", options.format])

        # Thumbnail
        if options.download_thumbnail:
            args.append("--write-thumbnail")

        # Subtítulos
        if options.download_subtitles:
            args.append("--write-subs")
            args.append("--write-auto-subs")
            args.extend(["--sub-langs", ",".join(options.subtitle_languages)])

        # Cookies
        if options.cookies_file and options.cookies_file.exists():
            args.extend(["--cookies", str(options.cookies_file)])

        # Proxy
        if options.proxy:
            args.extend(["--proxy", options.proxy])

        # Rate limit
        if options.rate_limit:
            args.extend(["--limit-rate", options.rate_limit])

        # Retries y timeout
        args.extend(["--retries", str(options.retries)])
        args.extend(["--socket-timeout", str(options.timeout)])

        # Evitar errores comunes
        args.extend([
            "--no-check-certificate",
            "--ignore-errors",
            "--no-warnings",
            "--no-part", # Evitar archivos .part para descargas más rápidas en sistemas estables
        ])

        # URL al final
        args.append(url)

        return args

    async def download(
        self,
        url: str,
        custom_filename: Optional[str] = None,
        options: Optional[DownloadOptions] = None,
    ) -> Optional[ClipInfo]:
        """
        Descarga un clip de la URL especificada.

        Args:
            url: URL del clip
            custom_filename: Nombre personalizado (sin extensión)
            options: Opciones de descarga (usa defaults si no se especifica)

        Returns:
            ClipInfo con la información del clip descargado, o None si falla
        """
        opts = options or self.options
        source = self._detect_source(url)

        self._stats["total_downloads"] += 1

        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_filename:
            base_filename = custom_filename
        else:
            base_filename = f"{source.value}_{timestamp}"

        output_template = str(self.output_dir / f"{base_filename}.%(ext)s")

        # Construir comando
        args = self._build_ytdlp_args(url, output_template, opts)

        try:
            # Ejecutar yt-dlp como proceso
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=opts.timeout * 2  # Doble del timeout por seguridad
            )

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self._stats["failed"] += 1
                return None

            # Buscar archivo descargado
            clip_info = await self._find_downloaded_file(base_filename, url, source)

            if clip_info:
                self._stats["successful"] += 1
                if clip_info.file_path.exists():
                    self._stats["total_bytes"] += clip_info.file_path.stat().st_size

            return clip_info

        except asyncio.TimeoutError:
            self._stats["failed"] += 1
            return None

        except Exception as e:
            self._stats["failed"] += 1
            return None

    async def _find_downloaded_file(
        self,
        base_filename: str,
        url: str,
        source: ClipSource,
    ) -> Optional[ClipInfo]:
        """Busca el archivo descargado y extrae metadata."""
        # Buscar archivo de video
        video_extensions = [".mp4", ".webm", ".mkv", ".mov", ".avi"]
        video_file = None

        for ext in video_extensions:
            potential_file = self.output_dir / f"{base_filename}{ext}"
            if potential_file.exists():
                video_file = potential_file
                break

        # Si no encontramos con el nombre exacto, buscar por patrón
        if not video_file:
            for file in self.output_dir.glob(f"{base_filename}*"):
                if file.suffix.lower() in video_extensions:
                    video_file = file
                    break

        if not video_file:
            return None

        # Buscar archivo de metadata JSON
        info_file = video_file.with_suffix(".info.json")
        metadata = {}
        title = base_filename
        duration = 0.0
        clip_id = base_filename

        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    title = metadata.get("title", base_filename)
                    duration = float(metadata.get("duration", 0))
                    clip_id = metadata.get("id", base_filename)
            except Exception:
                pass

        # Buscar thumbnail
        thumbnail_file = None
        thumb_extensions = [".jpg", ".png", ".webp"]
        for ext in thumb_extensions:
            potential_thumb = video_file.with_suffix(ext)
            if potential_thumb.exists():
                thumbnail_file = potential_thumb
                break

        return ClipInfo(
            clip_id=clip_id,
            source=source,
            url=url,
            title=title,
            duration=duration,
            file_path=video_file,
            thumbnail_path=thumbnail_file,
            metadata=metadata,
        )

    async def download_multiple(
        self,
        urls: List[str],
        max_concurrent: int = 3,
    ) -> List[ClipInfo]:
        """
        Descarga múltiples clips en paralelo.

        Args:
            urls: Lista de URLs
            max_concurrent: Máximo de descargas simultáneas

        Returns:
            Lista de ClipInfo (solo los exitosos)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(url: str) -> Optional[ClipInfo]:
            async with semaphore:
                return await self.download(url)

        tasks = [download_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)

        return [clip for clip in results if clip is not None]

    async def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un video sin descargarlo.
        Usa caché en memoria.
        
        Args:
            url: URL del video

        Returns:
            Diccionario con metadata o None si falla
        """
        # Verificar caché
        if url in self._info_cache:
            return self._info_cache[url]

        args = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30
            )

            if process.returncode == 0 and stdout:
                info = json.loads(stdout.decode())
                # Guardar en caché (limitar tamaño si fuera necesario)
                self._info_cache[url] = info
                return info

            return None

        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de descargas."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful"] / self._stats["total_downloads"]
                if self._stats["total_downloads"] > 0 else 0
            ),
            "total_mb": self._stats["total_bytes"] / (1024 * 1024),
        }

    def cleanup_temp(self) -> int:
        """Limpia archivos temporales. Retorna cantidad eliminada."""
        count = 0
        if self.temp_dir.exists():
            for file in self.temp_dir.iterdir():
                try:
                    file.unlink()
                    count += 1
                except Exception:
                    pass
        return count


# =====================================================
# CAPTURA DE STREAMS EN VIVO
# =====================================================

async def capture_live_stream(
    channel: str,
    output_dir: Path,
    duration: int = 30,
    platform: str = "kick",
) -> Optional[ClipInfo]:
    """
    Captura un segmento de un stream EN VIVO.

    Args:
        channel: Nombre del canal (slug)
        output_dir: Directorio de salida
        duration: Duración en segundos a capturar
        platform: Plataforma (kick, twitch)

    Returns:
        ClipInfo o None
    """
    from core.logger import logger

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{platform}_{channel}_{timestamp}"
    output_path = output_dir / f"{filename}.mp4"

    # URL del stream
    if platform == "kick":
        stream_url = f"https://kick.com/{channel}"
    elif platform == "twitch":
        stream_url = f"https://twitch.tv/{channel}"
    else:
        stream_url = channel  # Asumir URL directa

    # Buscar yt-dlp en el venv
    venv_ytdlp = Path(__file__).parent.parent / ".venv" / "Scripts" / "yt-dlp.exe"
    ytdlp_cmd = str(venv_ytdlp) if venv_ytdlp.exists() else "yt-dlp"

    # FFmpeg location
    ffmpeg_location = None
    ffmpeg_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for p in ffmpeg_path.glob("Gyan.FFmpeg*/ffmpeg-*/bin"):
        if (p / "ffmpeg.exe").exists():
            ffmpeg_location = str(p)
            break

    # Construir comando para capturar stream en vivo
    args = [
        ytdlp_cmd,
        "-f", "best",
        "--download-sections", f"*0-{duration}",
        "-o", str(output_path),
        "--no-part",
        "--no-playlist",
        "--socket-timeout", "30",
    ]

    if ffmpeg_location:
        args.extend(["--ffmpeg-location", ffmpeg_location])

    args.append(stream_url)

    logger.info(f"[Capture] Iniciando captura de {channel} ({duration}s)...")
    logger.debug(f"[Capture] Comando: {' '.join(args)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Timeout generoso para streams en vivo
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=duration + 120  # Duración + 2 minutos extra
        )

        if process.returncode != 0:
            error = stderr.decode() if stderr else "Error desconocido"
            logger.error(f"[Capture] Error: {error[:200]}")
            return None

        # Verificar que el archivo existe
        if output_path.exists():
            logger.info(f"[Capture] OK: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

            return ClipInfo(
                clip_id=filename,
                source=ClipSource.KICK if platform == "kick" else ClipSource.TWITCH,
                url=stream_url,
                title=f"Live capture - {channel}",
                duration=duration,
                file_path=output_path,
                metadata={"channel": channel, "platform": platform, "live_capture": True},
            )
        else:
            logger.error(f"[Capture] Archivo no creado: {output_path}")
            return None

    except asyncio.TimeoutError:
        logger.error(f"[Capture] Timeout capturando {channel}")
        return None
    except Exception as e:
        logger.error(f"[Capture] Exception: {e}")
        return None


# =====================================================
# FUNCIONES HELPER PARA KICK
# =====================================================

async def download_kick_clip(
    clip_url: str,
    output_dir: Path,
    channel: Optional[str] = None,
) -> Optional[ClipInfo]:
    """
    Helper para descargar un clip de Kick.

    Args:
        clip_url: URL del clip de Kick
        output_dir: Directorio de salida
        channel: Nombre del canal (opcional, para el nombre del archivo)

    Returns:
        ClipInfo o None
    """
    downloader = ClipDownloader(output_dir=output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kick_{channel}_{timestamp}" if channel else None

    return await downloader.download(clip_url, custom_filename=filename)


async def download_kick_vod_segment(
    vod_url: str,
    output_dir: Path,
    start_time: str,  # formato: "HH:MM:SS" o segundos
    duration: str,    # formato: "HH:MM:SS" o segundos
) -> Optional[ClipInfo]:
    """
    Descarga un segmento de un VOD de Kick.

    Args:
        vod_url: URL del VOD
        output_dir: Directorio de salida
        start_time: Tiempo de inicio
        duration: Duración del segmento

    Returns:
        ClipInfo o None
    """
    # yt-dlp soporta --download-sections para esto
    downloader = ClipDownloader(output_dir=output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_template = str(output_dir / f"kick_segment_{timestamp}.%(ext)s")

    args = [
        "yt-dlp",
        "--download-sections", f"*{start_time}-{duration}",
        "-o", output_template,
        "--write-info-json",
        "-f", "best",
        vod_url,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await asyncio.wait_for(process.communicate(), timeout=300)

        if process.returncode == 0:
            return await downloader._find_downloaded_file(
                f"kick_segment_{timestamp}",
                vod_url,
                ClipSource.KICK,
            )

        return None

    except Exception:
        return None


# =====================================================
# EJEMPLO DE USO
# =====================================================

async def main():
    """Ejemplo de uso del ClipDownloader."""
    from core.logger import logger

    output_dir = Path("./clips")
    downloader = ClipDownloader(output_dir=output_dir)

    # Ejemplo: descargar un clip
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Ejemplo

    logger.info(f"Obteniendo info de: {test_url}")
    info = await downloader.get_video_info(test_url)

    if info:
        logger.info(f"Título: {info.get('title')}")
        logger.info(f"Duración: {info.get('duration')}s")
        logger.info(f"Canal: {info.get('uploader')}")

    # Descomentar para descargar
    # clip = await downloader.download(test_url)
    # if clip:
    #     logger.info(f"Descargado: {clip.file_path}")

    logger.info(f"Stats: {downloader.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
