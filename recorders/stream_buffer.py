"""
Buffer circular para captura de streams en vivo.

Mantiene un buffer de video que permite capturar los segundos
ANTES de que ocurra un evento (spike de chat, etc.).

Arquitectura:
- Un proceso FFmpeg captura continuamente el stream a segmentos HLS
- Los segmentos se mantienen en un directorio temporal (buffer)
- Al detectar un evento, se concatenan los segmentos relevantes
"""

import asyncio
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Awaitable
from dataclasses import dataclass, field
import subprocess
import tempfile
import glob


@dataclass
class BufferSegment:
    """Información de un segmento de video."""
    file_path: Path
    start_time: datetime
    duration: float  # segundos
    sequence: int


@dataclass
class BufferConfig:
    """Configuración del buffer."""
    buffer_seconds: int = 60  # Mantener 60 segundos en buffer
    segment_duration: int = 5  # Cada segmento dura 5 segundos
    pre_event_seconds: int = 10  # Capturar 10 segundos antes del evento
    post_event_seconds: int = 20  # Capturar 20 segundos después


class StreamBuffer:
    """
    Buffer circular para streams en vivo.

    Mantiene un buffer continuo del stream que permite
    capturar clips incluyendo segundos ANTES de un evento.

    Uso:
        buffer = StreamBuffer(channel="roshtein", platform="kick")
        await buffer.start()

        # Cuando detectamos spike...
        clip_path = await buffer.capture_clip(
            pre_seconds=10,  # 10 segundos antes
            post_seconds=20  # 20 segundos después
        )
    """

    def __init__(
        self,
        channel: str,
        platform: str = "kick",
        config: Optional[BufferConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        self.channel = channel
        self.platform = platform
        self.config = config or BufferConfig()

        # Directorio base
        self.output_dir = output_dir or Path("clips/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Directorio temporal para segmentos
        self._temp_dir = Path(tempfile.mkdtemp(prefix=f"buffer_{channel}_"))

        # Estado
        self._is_running = False
        self._ffmpeg_process: Optional[asyncio.subprocess.Process] = None
        self._segments: List[BufferSegment] = []
        self._sequence = 0
        self._cleanup_task: Optional[asyncio.Task] = None

        # FFmpeg path
        self._ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """Encuentra la ruta de FFmpeg."""
        # Buscar en winget installation
        ffmpeg_base = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        for p in ffmpeg_base.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"):
            return str(p)

        # Fallback a PATH
        return "ffmpeg"

    def _get_stream_url(self) -> str:
        """Obtiene la URL del stream."""
        if self.platform == "kick":
            return f"https://kick.com/{self.channel}"
        elif self.platform == "twitch":
            return f"https://twitch.tv/{self.channel}"
        return self.channel

    async def start(self) -> bool:
        """
        Inicia la captura del buffer.

        Returns:
            True si inició correctamente
        """
        if self._is_running:
            return True

        try:
            # Primero obtenemos la URL real del stream con yt-dlp
            stream_url = await self._get_manifest_url()
            if not stream_url:
                return False

            # Iniciar FFmpeg capturando a segmentos HLS
            segment_pattern = str(self._temp_dir / "segment_%05d.ts")

            cmd = [
                self._ffmpeg_path,
                "-i", stream_url,
                "-c", "copy",  # Sin re-encoding
                "-f", "segment",
                "-segment_time", str(self.config.segment_duration),
                "-reset_timestamps", "1",
                "-segment_format", "mpegts",
                segment_pattern,
            ]

            self._ffmpeg_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            self._is_running = True

            # Iniciar tarea de limpieza/monitoreo de segmentos
            self._cleanup_task = asyncio.create_task(self._monitor_segments())

            return True

        except Exception as e:
            print(f"[Buffer] Error starting: {e}")
            return False

    async def _get_manifest_url(self) -> Optional[str]:
        """Obtiene la URL del manifest HLS usando yt-dlp."""
        venv_ytdlp = Path(__file__).parent.parent / ".venv" / "Scripts" / "yt-dlp.exe"
        ytdlp_cmd = str(venv_ytdlp) if venv_ytdlp.exists() else "yt-dlp"

        stream_url = self._get_stream_url()

        cmd = [
            ytdlp_cmd,
            "-g",  # Solo obtener URL
            "-f", "best",
            stream_url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=30
            )

            if process.returncode == 0 and stdout:
                return stdout.decode().strip()
            return None

        except Exception:
            return None

    async def _monitor_segments(self):
        """Monitorea y limpia segmentos antiguos."""
        while self._is_running:
            try:
                # Buscar nuevos segmentos
                segment_files = sorted(self._temp_dir.glob("segment_*.ts"))

                for seg_file in segment_files:
                    # Verificar si ya está registrado
                    if not any(s.file_path == seg_file for s in self._segments):
                        self._sequence += 1
                        self._segments.append(BufferSegment(
                            file_path=seg_file,
                            start_time=datetime.now(),
                            duration=self.config.segment_duration,
                            sequence=self._sequence,
                        ))

                # Limpiar segmentos antiguos (fuera del buffer)
                max_segments = self.config.buffer_seconds // self.config.segment_duration
                while len(self._segments) > max_segments:
                    old_segment = self._segments.pop(0)
                    try:
                        old_segment.file_path.unlink()
                    except Exception:
                        pass

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Buffer] Monitor error: {e}")
                await asyncio.sleep(1)

    async def stop(self):
        """Detiene la captura del buffer."""
        self._is_running = False

        # Cancelar tarea de monitoreo
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Terminar FFmpeg
        if self._ffmpeg_process:
            self._ffmpeg_process.terminate()
            try:
                await asyncio.wait_for(
                    self._ffmpeg_process.wait(),
                    timeout=5
                )
            except asyncio.TimeoutError:
                self._ffmpeg_process.kill()

        # Limpiar directorio temporal
        try:
            shutil.rmtree(self._temp_dir)
        except Exception:
            pass

    async def capture_clip(
        self,
        pre_seconds: Optional[int] = None,
        post_seconds: Optional[int] = None,
        output_filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Captura un clip del buffer.

        Args:
            pre_seconds: Segundos antes del momento actual a incluir
            post_seconds: Segundos después a esperar y capturar
            output_filename: Nombre del archivo de salida (sin extensión)

        Returns:
            Path del clip generado o None
        """
        pre_secs = pre_seconds or self.config.pre_event_seconds
        post_secs = post_seconds or self.config.post_event_seconds

        # Calcular cuántos segmentos necesitamos del pasado
        segments_needed = pre_secs // self.config.segment_duration + 1

        # Obtener segmentos del buffer (los más recientes)
        pre_segments = self._segments[-segments_needed:] if segments_needed <= len(self._segments) else self._segments.copy()

        # Esperar post_seconds para capturar el "después"
        if post_secs > 0:
            await asyncio.sleep(post_secs)

        # Obtener segmentos adicionales post-evento
        post_segments_count = post_secs // self.config.segment_duration + 1
        all_segments = pre_segments + self._segments[-post_segments_count:]

        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_segments = []
        for seg in all_segments:
            if seg.sequence not in seen:
                seen.add(seg.sequence)
                unique_segments.append(seg)

        if not unique_segments:
            return None

        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_filename or f"{self.platform}_{self.channel}_{timestamp}"
        output_path = self.output_dir / f"{filename}.mp4"

        # Concatenar segmentos con FFmpeg
        return await self._concat_segments(unique_segments, output_path)

    async def _concat_segments(
        self,
        segments: List[BufferSegment],
        output_path: Path,
    ) -> Optional[Path]:
        """Concatena segmentos en un solo archivo."""
        if not segments:
            return None

        # Crear archivo de lista para FFmpeg concat
        list_file = self._temp_dir / "concat_list.txt"
        with open(list_file, "w") as f:
            for seg in sorted(segments, key=lambda s: s.sequence):
                if seg.file_path.exists():
                    f.write(f"file '{seg.file_path}'\n")

        # Concatenar con FFmpeg
        cmd = [
            self._ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
            "-y",  # Sobrescribir si existe
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await asyncio.wait_for(
                process.communicate(),
                timeout=60
            )

            if process.returncode == 0 and output_path.exists():
                return output_path
            return None

        except Exception as e:
            print(f"[Buffer] Concat error: {e}")
            return None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def buffer_duration(self) -> float:
        """Duración actual del buffer en segundos."""
        return len(self._segments) * self.config.segment_duration

    def get_status(self) -> Dict:
        """Retorna el estado del buffer."""
        return {
            "channel": self.channel,
            "platform": self.platform,
            "is_running": self._is_running,
            "buffer_duration": self.buffer_duration,
            "segments_count": len(self._segments),
            "temp_dir": str(self._temp_dir),
        }


class MultiStreamBuffer:
    """
    Gestor de múltiples buffers de stream.

    Mantiene un buffer activo por cada canal monitoreado.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("clips/raw")
        self._buffers: Dict[str, StreamBuffer] = {}
        self._lock = asyncio.Lock()

    async def add_channel(
        self,
        channel: str,
        platform: str = "kick",
        config: Optional[BufferConfig] = None,
    ) -> bool:
        """Agrega un canal al buffer."""
        async with self._lock:
            key = f"{platform}:{channel}"
            if key in self._buffers:
                return True

            buffer = StreamBuffer(
                channel=channel,
                platform=platform,
                config=config,
                output_dir=self.output_dir,
            )

            success = await buffer.start()
            if success:
                self._buffers[key] = buffer
            return success

    async def remove_channel(self, channel: str, platform: str = "kick"):
        """Remueve un canal del buffer."""
        async with self._lock:
            key = f"{platform}:{channel}"
            if key in self._buffers:
                await self._buffers[key].stop()
                del self._buffers[key]

    async def capture_clip(
        self,
        channel: str,
        platform: str = "kick",
        pre_seconds: int = 10,
        post_seconds: int = 20,
    ) -> Optional[Path]:
        """Captura un clip de un canal específico."""
        key = f"{platform}:{channel}"
        if key not in self._buffers:
            return None

        return await self._buffers[key].capture_clip(
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
        )

    async def stop_all(self):
        """Detiene todos los buffers."""
        async with self._lock:
            for buffer in self._buffers.values():
                await buffer.stop()
            self._buffers.clear()

    def get_status(self) -> Dict:
        """Retorna el estado de todos los buffers."""
        return {
            key: buffer.get_status()
            for key, buffer in self._buffers.items()
        }
