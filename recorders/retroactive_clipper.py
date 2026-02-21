"""
Retroactive Clipper - Algoritmo Divide y Vencerás

En lugar de mantener un buffer constante en RAM, este sistema:
1. Registra timestamps de spikes detectados
2. Cuando hay spike, descarga el segmento del stream (últimos N minutos)
3. Extrae solo el clip relevante (5 seg antes + spike + 20 seg después)
4. Descarta el resto

Ventajas:
- No consume RAM/disco constantemente
- Solo descarga cuando hay spike real
- Usa los timestamps exactos del WebSocket
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import json
import os


@dataclass
class SpikeEvent:
    """Representa un spike detectado."""
    channel: str
    timestamp: datetime
    velocity: float
    viewers: int = 0
    processed: bool = False
    clip_path: Optional[Path] = None


@dataclass
class RetroClipConfig:
    """Configuración del clipper retroactivo."""
    pre_spike_seconds: int = 5      # Segundos ANTES del spike
    post_spike_seconds: int = 25    # Segundos DESPUÉS del spike
    segment_duration: int = 60      # Duración del segmento a descargar (segundos)
    min_spike_interval: int = 30    # Mínimo intervalo entre clips del mismo canal
    max_pending_spikes: int = 10    # Máximo de spikes pendientes por canal


class RetroactiveClipper:
    """
    Sistema de clips retroactivos usando divide y vencerás.

    Flujo:
    1. Monitor detecta spike → registra timestamp
    2. Clipper descarga segmento del stream (últimos 60 seg)
    3. Extrae clip preciso: [spike - 5s, spike + 25s]
    4. Retorna path del clip para procesamiento

    Ejemplo:
        clipper = RetroactiveClipper(output_dir=Path("./clips/raw"))

        # Cuando se detecta spike
        clipper.register_spike("westcol", velocity=150.0)

        # Procesar spikes pendientes
        clips = await clipper.process_pending_spikes()
    """

    def __init__(
        self,
        output_dir: Path = Path("./clips/raw"),
        config: Optional[RetroClipConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or RetroClipConfig()

        # Spikes pendientes por canal
        self._pending_spikes: Dict[str, List[SpikeEvent]] = {}

        # Último clip por canal (para cooldown)
        self._last_clip_time: Dict[str, datetime] = {}

        # Lock para operaciones concurrentes
        self._lock = asyncio.Lock()

        # Buscar herramientas
        self._ytdlp_path = self._find_ytdlp()
        self._ffmpeg_path = self._find_ffmpeg()

    def _find_ytdlp(self) -> str:
        """Encuentra yt-dlp en el sistema."""
        venv_path = Path(__file__).parent.parent / ".venv" / "Scripts" / "yt-dlp.exe"
        if venv_path.exists():
            return str(venv_path)
        return "yt-dlp"

    def _find_ffmpeg(self) -> Optional[str]:
        """Encuentra ffmpeg en el sistema."""
        # Buscar en WinGet packages
        for p in (Path.home() / "AppData/Local/Microsoft/WinGet/Packages").glob("Gyan.FFmpeg*/ffmpeg-*/bin"):
            if (p / "ffmpeg.exe").exists():
                return str(p)
        return None

    def register_spike(
        self,
        channel: str,
        velocity: float,
        viewers: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Registra un spike para procesamiento posterior.

        Args:
            channel: Nombre del canal
            velocity: Velocidad del chat (msgs/seg)
            viewers: Número de viewers
            timestamp: Momento del spike (default: ahora)

        Returns:
            True si se registró, False si está en cooldown
        """
        now = timestamp or datetime.now()

        # Verificar cooldown
        last_clip = self._last_clip_time.get(channel)
        if last_clip:
            elapsed = (now - last_clip).total_seconds()
            if elapsed < self.config.min_spike_interval:
                print(f"[RetroClipper] {channel}: En cooldown ({elapsed:.0f}s < {self.config.min_spike_interval}s)")
                return False

        # Crear evento
        spike = SpikeEvent(
            channel=channel,
            timestamp=now,
            velocity=velocity,
            viewers=viewers,
        )

        # Agregar a pendientes
        if channel not in self._pending_spikes:
            self._pending_spikes[channel] = []

        # Limitar cantidad de pendientes
        if len(self._pending_spikes[channel]) >= self.config.max_pending_spikes:
            self._pending_spikes[channel].pop(0)  # Remover el más viejo

        self._pending_spikes[channel].append(spike)
        print(f"[RetroClipper] {channel}: Spike registrado (velocity={velocity:.1f}, viewers={viewers})")

        return True

    async def process_pending_spikes(self) -> List[Tuple[SpikeEvent, Path]]:
        """
        Procesa todos los spikes pendientes.

        Returns:
            Lista de (spike, clip_path) procesados exitosamente
        """
        results = []

        async with self._lock:
            for channel, spikes in list(self._pending_spikes.items()):
                # Procesar spikes no procesados
                for spike in spikes:
                    if spike.processed:
                        continue

                    clip_path = await self._process_spike(spike)

                    if clip_path:
                        spike.processed = True
                        spike.clip_path = clip_path
                        self._last_clip_time[channel] = datetime.now()
                        results.append((spike, clip_path))

                # Limpiar spikes procesados
                self._pending_spikes[channel] = [s for s in spikes if not s.processed]

        return results

    async def process_spike_immediate(
        self,
        channel: str,
        velocity: float,
        viewers: int = 0,
    ) -> Optional[Path]:
        """
        Procesa un spike inmediatamente (sin cola).

        Args:
            channel: Nombre del canal
            velocity: Velocidad del chat
            viewers: Número de viewers

        Returns:
            Path del clip o None
        """
        spike = SpikeEvent(
            channel=channel,
            timestamp=datetime.now(),
            velocity=velocity,
            viewers=viewers,
        )

        return await self._process_spike(spike)

    async def _process_spike(self, spike: SpikeEvent) -> Optional[Path]:
        """
        Procesa un spike individual: descarga segmento y extrae clip.

        Algoritmo Divide y Vencerás:
        1. Descargar segmento amplio del stream (60 seg)
        2. Calcular offset del spike dentro del segmento
        3. Extraer clip preciso con ffmpeg
        """
        channel = spike.channel
        timestamp = spike.timestamp

        print(f"[RetroClipper] {channel}: Procesando spike de {timestamp.strftime('%H:%M:%S')}")

        try:
            # Paso 1: Descargar segmento del stream
            segment_path = await self._download_segment(channel, self.config.segment_duration)

            if not segment_path or not segment_path.exists():
                print(f"[RetroClipper] {channel}: Error descargando segmento")
                return None

            print(f"[RetroClipper] {channel}: Segmento descargado ({segment_path.stat().st_size / 1024 / 1024:.1f} MB)")

            # Paso 2: El spike está al final del segmento (acabamos de descargarlo)
            # Calculamos el punto de corte
            segment_duration = await self._get_video_duration(segment_path)

            if segment_duration is None:
                print(f"[RetroClipper] {channel}: No se pudo obtener duración")
                # Usar el segmento completo
                return segment_path

            # El spike ocurrió hace ~0 segundos (acabamos de detectarlo)
            # El segmento contiene los últimos N segundos
            # Queremos: [final - pre_spike - post_spike, final]

            # Punto de inicio del clip dentro del segmento
            clip_duration = self.config.pre_spike_seconds + self.config.post_spike_seconds
            start_offset = max(0, segment_duration - clip_duration)

            # Paso 3: Extraer clip con ffmpeg
            clip_path = await self._extract_clip(
                segment_path,
                start_offset,
                clip_duration,
                channel,
            )

            # Limpiar segmento temporal
            try:
                segment_path.unlink()
            except:
                pass

            if clip_path and clip_path.exists():
                print(f"[RetroClipper] {channel}: Clip extraído ({clip_path.stat().st_size / 1024 / 1024:.1f} MB)")
                return clip_path

            return None

        except Exception as e:
            print(f"[RetroClipper] {channel}: Error procesando spike: {e}")
            return None

    async def _download_segment(self, channel: str, duration: int) -> Optional[Path]:
        """Descarga un segmento del stream en vivo usando subprocess síncrono."""
        import subprocess

        url = f"https://kick.com/{channel}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"segment_{channel}_{timestamp}.mp4"

        # IMPORTANTE: Usar ruta absoluta para el output
        absolute_output = output_path.resolve()

        # Construir comando como lista
        cmd = [
            self._ytdlp_path,
            "-f", "b",
            "--download-sections", f"*0-{duration}",
            "-o", str(absolute_output),
            "--no-playlist",
            "--progress",
            url
        ]

        if self._ffmpeg_path:
            cmd.insert(1, "--ffmpeg-location")
            cmd.insert(2, self._ffmpeg_path)

        print(f"[RetroClipper] {channel}: Iniciando descarga de {duration}s...")
        print(f"[RetroClipper] {channel}: URL: {url}")
        print(f"[RetroClipper] {channel}: Output: {absolute_output}")

        try:
            # Usar run_in_executor para no bloquear el event loop
            loop = asyncio.get_event_loop()

            def run_ytdlp():
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=duration + 120
                )
                return result

            result = await loop.run_in_executor(None, run_ytdlp)

            # Mostrar output para debug
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print(f"[RetroClipper] {channel}: stdout: {lines[-1] if lines else '(vacío)'}")

            if result.stderr:
                lines = result.stderr.strip().split('\n')
                print(f"[RetroClipper] {channel}: stderr: {lines[-1][:200] if lines else '(vacío)'}")

            if result.returncode != 0:
                print(f"[RetroClipper] {channel}: yt-dlp exit code: {result.returncode}")

            if absolute_output.exists() and absolute_output.stat().st_size > 0:
                size_mb = absolute_output.stat().st_size / 1024 / 1024
                print(f"[RetroClipper] {channel}: Descarga exitosa! ({size_mb:.1f} MB)")
                return absolute_output

            print(f"[RetroClipper] {channel}: Archivo no existe o está vacío en {absolute_output}")
            return None

        except subprocess.TimeoutExpired:
            print(f"[RetroClipper] {channel}: Timeout descargando segmento ({duration + 120}s)")
            return None
        except Exception as e:
            import traceback
            print(f"[RetroClipper] {channel}: Error en descarga: {e}")
            print(f"[RetroClipper] {channel}: Traceback: {traceback.format_exc()}")
            return None

    async def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """Obtiene la duración de un video con ffprobe."""
        ffprobe = "ffprobe"
        if self._ffmpeg_path:
            ffprobe = str(Path(self._ffmpeg_path) / "ffprobe.exe")

        cmd = [
            ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await process.communicate()

            if process.returncode == 0:
                data = json.loads(stdout.decode())
                duration = float(data.get("format", {}).get("duration", 0))
                return duration if duration > 0 else None

            return None

        except Exception:
            return None

    async def _extract_clip(
        self,
        source_path: Path,
        start_offset: float,
        duration: float,
        channel: str,
    ) -> Optional[Path]:
        """Extrae un clip de un video con ffmpeg."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{channel}_{timestamp}.mp4"

        ffmpeg = "ffmpeg"
        if self._ffmpeg_path:
            ffmpeg = str(Path(self._ffmpeg_path) / "ffmpeg.exe")

        cmd = [
            ffmpeg,
            "-y",
            "-ss", str(start_offset),
            "-i", str(source_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await process.communicate()

            if output_path.exists() and output_path.stat().st_size > 0:
                return output_path

            return None

        except Exception as e:
            print(f"[RetroClipper] Error extrayendo clip: {e}")
            return None

    def get_pending_count(self) -> Dict[str, int]:
        """Retorna cantidad de spikes pendientes por canal."""
        return {
            channel: len([s for s in spikes if not s.processed])
            for channel, spikes in self._pending_spikes.items()
        }

    def clear_pending(self, channel: Optional[str] = None):
        """Limpia spikes pendientes."""
        if channel:
            self._pending_spikes.pop(channel, None)
        else:
            self._pending_spikes.clear()


# ============================================
# INTEGRACIÓN CON PIPELINE EXISTENTE
# ============================================

class SmartClipper:
    """
    Clipper inteligente que decide el mejor método:
    - Si el stream está en vivo → RetroactiveClipper (divide y vencerás)
    - Si es un VOD → Descarga directa

    Usa los timestamps del WebSocket para precisión máxima.
    """

    def __init__(self, output_dir: Path = Path("./clips/raw")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.retro_clipper = RetroactiveClipper(
            output_dir=output_dir,
            config=RetroClipConfig(
                pre_spike_seconds=5,
                post_spike_seconds=25,
                segment_duration=45,  # 45 segundos de segmento
                min_spike_interval=30,
            )
        )

        # Cache de streams en vivo
        self._live_streams: Dict[str, bool] = {}

    def set_stream_live(self, channel: str, is_live: bool):
        """Marca un stream como en vivo o no."""
        self._live_streams[channel] = is_live

    async def capture_spike(
        self,
        channel: str,
        velocity: float,
        viewers: int = 0,
        force_retro: bool = True,
    ) -> Optional[Path]:
        """
        Captura un clip de un spike detectado.

        Args:
            channel: Nombre del canal
            velocity: Velocidad del chat
            viewers: Número de viewers
            force_retro: Forzar método retroactivo

        Returns:
            Path del clip o None
        """
        is_live = self._live_streams.get(channel, True)

        if is_live or force_retro:
            # Usar método retroactivo (divide y vencerás)
            return await self.retro_clipper.process_spike_immediate(
                channel=channel,
                velocity=velocity,
                viewers=viewers,
            )
        else:
            # Stream no está en vivo, no podemos capturar
            print(f"[SmartClipper] {channel}: Stream no está en vivo")
            return None

    def register_spike(self, channel: str, velocity: float, viewers: int = 0) -> bool:
        """Registra spike para procesamiento en batch."""
        return self.retro_clipper.register_spike(channel, velocity, viewers)

    async def process_batch(self) -> List[Tuple[SpikeEvent, Path]]:
        """Procesa todos los spikes pendientes."""
        return await self.retro_clipper.process_pending_spikes()


# ============================================
# TEST
# ============================================

async def test_retroactive_clipper():
    """Prueba del clipper retroactivo."""
    clipper = RetroactiveClipper(
        output_dir=Path("D:/proyecto stake/clips/raw"),
        config=RetroClipConfig(
            pre_spike_seconds=5,
            post_spike_seconds=20,
            segment_duration=30,
        )
    )

    # Simular spike
    print("Simulando spike en westcol...")
    clip = await clipper.process_spike_immediate(
        channel="westcol",
        velocity=100.0,
        viewers=50000,
    )

    if clip:
        print(f"Clip creado: {clip}")
        print(f"Tamaño: {clip.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("No se pudo crear el clip")


if __name__ == "__main__":
    asyncio.run(test_retroactive_clipper())
