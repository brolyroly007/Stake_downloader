"""
Pipeline de procesamiento automático.

Conecta: Detección → Descarga → Procesamiento → Publicación

Cuando se detecta un momento viral:
1. Descarga el clip del stream
2. Lo procesa (reframe vertical, subtítulos)
3. Lo guarda para publicación
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import json

from monitors.base_monitor import MonitorEvent, EventType
from recorders.clip_downloader import ClipDownloader, DownloadOptions
import config

# Importar buffer de stream (legacy)
try:
    from recorders.stream_buffer import StreamBuffer, MultiStreamBuffer, BufferConfig
    HAS_BUFFER = True
except ImportError:
    HAS_BUFFER = False

# Importar clipper retroactivo (divide y vencerás) - NUEVO
try:
    from recorders.retroactive_clipper import SmartClipper, RetroClipConfig
    HAS_SMART_CLIPPER = True
    print("[Pipeline Import] SmartClipper cargado correctamente")
except ImportError as e:
    HAS_SMART_CLIPPER = False
    print(f"[Pipeline Import] SmartClipper no disponible: {e}")

# Importar analizador de Gemini
try:
    from detectors.gemini_analyzer import GeminiAnalyzer, ClipAnalysis, ViralityFilter
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Importar procesadores
try:
    from processors.reframe_vertical import VideoReframer, ReframeOptions, ReframeMode
    HAS_REFRAMER = True
except ImportError:
    HAS_REFRAMER = False

try:
    from processors.caption_generator import CaptionGenerator, CaptionOptions, CaptionStyle
    HAS_CAPTIONS = True
except ImportError:
    HAS_CAPTIONS = False


@dataclass
class ViralMoment:
    """Representa un momento viral detectado."""
    id: str
    event_type: str  # "spike"
    channel: str
    timestamp: datetime
    data: Dict[str, Any]

    # Estado del pipeline
    clip_url: Optional[str] = None
    clip_path: Optional[Path] = None          # Video raw descargado
    vertical_path: Optional[Path] = None      # Video vertical (9:16)
    captioned_path: Optional[Path] = None     # Video con subtítulos
    processed_path: Optional[Path] = None     # Video final procesado
    published_urls: Dict[str, str] = field(default_factory=dict)

    # AI Analysis
    ai_analysis: Optional[Any] = None  # ClipAnalysis from Gemini
    ai_score: float = 0.0
    ai_title: str = ""
    ai_tags: List[str] = field(default_factory=list)

    status: str = "detected"  # detected, downloading, analyzing, reframing, captioning, ready, published, failed
    error: Optional[str] = None


class ViralPipeline:
    """
    Pipeline automático para procesar momentos virales.

    Flujo completo:
    1. Recibe evento de detección (spike de chat)
    2. Descarga clip del stream en vivo
    3. Reencuadra a formato vertical (9:16)
    4. Genera subtítulos automáticos
    5. Marca como listo para publicar
    """

    def __init__(
        self,
        output_dir: Path = Path("./clips"),
        auto_download: bool = True,
        auto_process: bool = True,
        auto_upload: bool = False,
        clip_duration: int = 30,
        enable_buffer: bool = True,
        enable_ai_filter: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Crear subdirectorios
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "square").mkdir(exist_ok=True)  # Formato cuadrado 1:1
        (self.output_dir / "captioned").mkdir(exist_ok=True)
        (self.output_dir / "final").mkdir(exist_ok=True)

        self.auto_download = auto_download
        self.auto_process = auto_process
        self.auto_upload = auto_upload
        self.clip_duration = clip_duration
        self.enable_buffer = enable_buffer and HAS_BUFFER
        self.enable_ai_filter = enable_ai_filter and HAS_GEMINI

        # Almacén de momentos
        self.moments: List[ViralMoment] = []
        self.max_moments = 100

        # Downloader
        self.downloader = ClipDownloader(
            output_dir=self.output_dir / "raw",
            default_options=DownloadOptions(
                format="best",
                max_resolution=1080,
            )
        )

        # Smart Clipper (algoritmo divide y vencerás) - NUEVO
        self.smart_clipper: Optional[SmartClipper] = None
        if HAS_SMART_CLIPPER:
            self.smart_clipper = SmartClipper(output_dir=self.output_dir / "raw")
            print("[Pipeline] SmartClipper: OK (divide y vencerás)")
        else:
            print("[Pipeline] SmartClipper: NO DISPONIBLE")

        # Stream Buffer (legacy - para capturar ANTES del spike)
        self.multi_buffer: Optional[MultiStreamBuffer] = None
        if self.enable_buffer and not HAS_SMART_CLIPPER:
            buffer_config = BufferConfig(
                buffer_seconds=getattr(config, 'BUFFER_DURATION_SECONDS', 60),
                segment_duration=5,
                pre_event_seconds=getattr(config, 'PRE_SPIKE_SECONDS', 10),
                post_event_seconds=getattr(config, 'POST_SPIKE_SECONDS', 20),
            )
            self.multi_buffer = MultiStreamBuffer(output_dir=self.output_dir / "raw")
            print("[Pipeline] StreamBuffer: OK (legacy)")
        elif not HAS_SMART_CLIPPER:
            print("[Pipeline] StreamBuffer: NO DISPONIBLE")

        # Gemini AI Analyzer
        self.gemini_analyzer: Optional[GeminiAnalyzer] = None
        if self.enable_ai_filter:
            api_key = getattr(config, 'GEMINI_API_KEY', None)
            if api_key:
                self.gemini_analyzer = GeminiAnalyzer(api_key=api_key)
                print("[Pipeline] GeminiAnalyzer: OK")
            else:
                print("[Pipeline] GeminiAnalyzer: NO API KEY")
        else:
            print("[Pipeline] GeminiAnalyzer: DESHABILITADO")

        # Video Reframer (16:9 → 1:1 cuadrado)
        self.reframer = None
        if HAS_REFRAMER:
            self.reframer = VideoReframer(
                output_dir=self.output_dir / "square",
                default_options=ReframeOptions(
                    output_width=1080,
                    output_height=1080,  # Formato cuadrado 1:1
                    mode=ReframeMode.CENTER,  # Usar CENTER por defecto (más rápido)
                )
            )
            print("[Pipeline] VideoReframer: OK (1:1 cuadrado)")
        else:
            print("[Pipeline] VideoReframer: NO DISPONIBLE")

        # Caption Generator (subtítulos)
        self.captioner = None
        if HAS_CAPTIONS:
            self.captioner = CaptionGenerator(
                default_options=CaptionOptions(
                    model="base",  # tiny=rápido, base=balance, small/medium=preciso
                    style=CaptionStyle.TIKTOK,
                    font_size=48,
                    word_by_word=True,
                )
            )
            print("[Pipeline] CaptionGenerator: OK")
        else:
            print("[Pipeline] CaptionGenerator: NO DISPONIBLE (instalar whisper)")

        # Control de concurrencia
        self.max_concurrent_downloads = 3
        self._semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

        # Cola de procesamiento
        self._queue: asyncio.Queue = asyncio.Queue()
        self._processing = False

        # Callbacks
        self._on_moment_ready: List[callable] = []
        self._on_moment_analyzed: List[callable] = []  # Después de análisis AI

    def on_ready(self, callback):
        """Registra callback para cuando un momento está listo."""
        self._on_moment_ready.append(callback)

    def on_analyzed(self, callback):
        """Registra callback para cuando un momento es analizado por AI."""
        self._on_moment_analyzed.append(callback)

    async def handle_event(self, event: MonitorEvent) -> Optional[ViralMoment]:
        """
        Procesa un evento de detección.

        Args:
            event: Evento del monitor (spike de chat)

        Returns:
            ViralMoment creado o None
        """
        # Crear momento viral
        moment = ViralMoment(
            id=f"{event.channel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            event_type=event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
            channel=event.channel,
            timestamp=datetime.now(),
            data=event.data,
        )

        # Agregar a la lista
        self.moments.insert(0, moment)
        if len(self.moments) > self.max_moments:
            self.moments = self.moments[:self.max_moments]

        # Intentar obtener URL del clip
        clip_url = await self._get_clip_url(event)

        if clip_url:
            moment.clip_url = clip_url

            if self.auto_download:
                await self._queue.put(moment)

                # Iniciar procesador si no está corriendo
                if not self._processing:
                    asyncio.create_task(self._process_queue())

        return moment

    async def _get_clip_url(self, event: MonitorEvent) -> Optional[str]:
        """Obtiene la URL del clip para un evento."""
        channel = event.channel

        # Para eventos de Kick, construir URL del stream
        if event.source == "kick":
            return f"https://kick.com/{channel}"

        return None

    async def _process_queue(self):
        """Procesa la cola de momentos con concurrencia."""
        self._processing = True

        while not self._queue.empty():
            moment = await self._queue.get()
            
            # Lanzar tarea en background controlada por semáforo
            asyncio.create_task(self._process_moment_safe(moment))

        self._processing = False

    async def _process_moment_safe(self, moment: ViralMoment):
        """Wrapper para procesar momento con semáforo."""
        async with self._semaphore:
            await self._process_moment(moment)

    async def _process_moment(self, moment: ViralMoment):
        """Procesa un momento individual (descarga -> AI -> reframe -> caption)."""
        try:
            # ========== PASO 1: DESCARGAR ==========
            moment.status = "downloading"
            print(f"[Pipeline] {moment.id}: Descargando clip...")

            clip = await self._download_clip(moment)

            if not clip:
                moment.status = "failed"
                moment.error = "Download failed"
                print(f"[Pipeline] {moment.id}: Error en descarga")
                return

            moment.clip_path = clip
            print(f"[Pipeline] {moment.id}: Descarga OK ({clip.stat().st_size / 1024 / 1024:.1f} MB)")

            # ========== PASO 1.5: ANÁLISIS AI ==========
            if self.gemini_analyzer:
                moment.status = "analyzing"
                print(f"[Pipeline] {moment.id}: Analizando con Gemini AI...")

                ai_result = await self._analyze_with_ai(moment)

                if ai_result:
                    moment.ai_analysis = ai_result
                    moment.ai_score = ai_result.virality_score
                    moment.ai_title = ai_result.suggested_title
                    moment.ai_tags = ai_result.suggested_tags

                    print(f"[Pipeline] {moment.id}: AI Score={ai_result.virality_score:.2f}, Viral={ai_result.is_viral}")
                    print(f"[Pipeline] {moment.id}: Título sugerido: {ai_result.suggested_title}")

                    # Notificar callbacks de análisis
                    for callback in self._on_moment_analyzed:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(moment)
                            else:
                                callback(moment)
                        except Exception as e:
                            print(f"[Pipeline] Analyzed callback error: {e}")

                    # Filtrar si el score es muy bajo
                    min_score = getattr(config, 'AI_MIN_VIRALITY_SCORE', 0.6)
                    if ai_result.virality_score < min_score and not moment.data.get('manual_capture'):
                        print(f"[Pipeline] {moment.id}: Descartado por AI (score {ai_result.virality_score:.2f} < {min_score})")
                        moment.status = "filtered"
                        moment.error = f"AI score too low: {ai_result.virality_score:.2f}"

                        # ELIMINAR el clip descartado para no acumular basura
                        if moment.clip_path and moment.clip_path.exists():
                            try:
                                moment.clip_path.unlink()
                                print(f"[Pipeline] {moment.id}: Clip eliminado (liberando espacio)")
                            except Exception as e:
                                print(f"[Pipeline] {moment.id}: Error eliminando clip: {e}")

                        return

            # ========== PASO 2: REFRAME CUADRADO 1:1 ==========
            if self.auto_process and self.reframer:
                moment.status = "reframing"
                print(f"[Pipeline] {moment.id}: Convirtiendo a cuadrado 1:1...")

                square = await self._reframe_video(moment)

                if square:
                    moment.vertical_path = square  # Reutilizamos el campo
                    print(f"[Pipeline] {moment.id}: Reframe OK (1080x1080)")
                else:
                    print(f"[Pipeline] {moment.id}: Reframe falló, usando original")
                    moment.vertical_path = moment.clip_path

            # ========== PASO 3: SUBTÍTULOS ==========
            if self.auto_process and self.captioner:
                moment.status = "captioning"
                print(f"[Pipeline] {moment.id}: Generando subtítulos...")

                captioned = await self._add_captions(moment)

                if captioned:
                    moment.captioned_path = captioned
                    print(f"[Pipeline] {moment.id}: Subtítulos OK")
                else:
                    print(f"[Pipeline] {moment.id}: Subtítulos fallaron")

            # ========== DETERMINAR VIDEO FINAL ==========
            # Usar el video más procesado disponible
            moment.processed_path = (
                moment.captioned_path or
                moment.vertical_path or
                moment.clip_path
            )

            moment.status = "ready"
            print(f"[Pipeline] {moment.id}: LISTO - {moment.processed_path}")

            # Notificar callbacks
            for callback in self._on_moment_ready:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(moment)
                    else:
                        callback(moment)
                except Exception as e:
                    print(f"[Pipeline] Callback error: {e}")

        except Exception as e:
            moment.status = "failed"
            moment.error = str(e)
            print(f"[Pipeline] {moment.id}: Error - {e}")

    async def _download_clip(self, moment: ViralMoment) -> Optional[Path]:
        """Descarga el clip de un momento viral."""
        if not moment.clip_url:
            return None

        try:
            # Para streams en vivo de Kick - usar SmartClipper (divide y vencerás)
            if "kick.com" in moment.clip_url and self.smart_clipper:
                velocity = moment.data.get("velocity", 0)
                viewers = moment.data.get("viewers", 0)

                print(f"[Pipeline] {moment.id}: Usando SmartClipper (5s antes + 25s después)")
                clip_path = await self.smart_clipper.capture_spike(
                    channel=moment.channel,
                    velocity=velocity,
                    viewers=viewers,
                )

                if clip_path and clip_path.exists():
                    return clip_path

                # Fallback a método tradicional
                print(f"[Pipeline] {moment.id}: SmartClipper falló, usando método tradicional")
                return await self._download_live_stream(moment)

            # Para streams en vivo sin SmartClipper
            if "kick.com" in moment.clip_url:
                return await self._download_live_stream(moment)

            # Para otros, usar downloader normal
            clip = await self.downloader.download(
                moment.clip_url,
                custom_filename=moment.id,
            )

            if clip and clip.file_path.exists():
                return clip.file_path

        except Exception as e:
            print(f"[Pipeline] Download error: {e}")

        return None

    async def _download_live_stream(self, moment: ViralMoment) -> Optional[Path]:
        """Descarga un segmento de un stream en vivo."""
        output_path = self.output_dir / "raw" / f"{moment.id}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Buscar yt-dlp y ffmpeg
        venv_ytdlp = Path(__file__).parent / ".venv" / "Scripts" / "yt-dlp.exe"
        ytdlp_cmd = str(venv_ytdlp) if venv_ytdlp.exists() else "yt-dlp"

        ffmpeg_path = None
        for p in (Path.home() / "AppData/Local/Microsoft/WinGet/Packages").glob("Gyan.FFmpeg*/ffmpeg-*/bin"):
            if (p / "ffmpeg.exe").exists():
                ffmpeg_path = str(p)
                break

        args = [
            ytdlp_cmd,
            "-f", "b",
            "--download-sections", f"*0-{self.clip_duration}",
            "-o", str(output_path),
            "--no-playlist",
        ]

        if ffmpeg_path:
            args.extend(["--ffmpeg-location", ffmpeg_path])

        args.append(moment.clip_url)

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120  # 2 minutos máximo
            )

            if output_path.exists():
                return output_path

        except Exception as e:
            print(f"[Pipeline] Download stream error: {e}")

        return None

    async def _reframe_video(self, moment: ViralMoment) -> Optional[Path]:
        """Convierte el video a formato cuadrado 1:1."""
        if not self.reframer or not moment.clip_path:
            return None

        try:
            result = await self.reframer.reframe(
                moment.clip_path,
                output_filename=f"{moment.id}_square.mp4",
            )

            if result and result.output_path.exists():
                return result.output_path

        except Exception as e:
            print(f"[Pipeline] Reframe error: {e}")

        return None

    async def _add_captions(self, moment: ViralMoment) -> Optional[Path]:
        """Genera y añade subtítulos al video."""
        if not self.captioner:
            return None

        # Usar video vertical si existe, si no el original
        video_path = moment.vertical_path or moment.clip_path
        if not video_path:
            return None

        try:
            # Generar subtítulos
            result = await self.captioner.generate(
                video_path,
                output_dir=self.output_dir / "captioned",
            )

            if not result or not result.ass_path:
                return None

            # Incrustar subtítulos en el video
            output_path = self.output_dir / "final" / f"{moment.id}_final.mp4"

            success = await self.captioner.burn_captions(
                video_path,
                result.ass_path,
                output_path,
            )

            if success and output_path.exists():
                return output_path

        except Exception as e:
            print(f"[Pipeline] Caption error: {e}")

        return None

    async def _analyze_with_ai(self, moment: ViralMoment) -> Optional[Any]:
        """Analiza el clip con Gemini AI para determinar viralidad."""
        if not self.gemini_analyzer or not moment.clip_path:
            return None

        try:
            # Primero intentar transcribir el audio
            transcript = await self._transcribe_clip(moment.clip_path)

            if not transcript:
                print(f"[Pipeline] {moment.id}: No se pudo transcribir, usando contexto")
                # Usar solo el contexto del evento
                transcript = f"[Spike de chat detectado en {moment.channel}]"

            # Preparar contexto
            context = {
                "chat_velocity": moment.data.get("velocity", 0),
                "is_spike": True,
                "viewers": moment.data.get("viewers", 0),
            }

            # Analizar con Gemini
            analysis = await self.gemini_analyzer.analyze_transcript(
                transcript=transcript,
                channel=moment.channel,
                context=context,
            )

            return analysis

        except Exception as e:
            print(f"[Pipeline] AI analysis error: {e}")
            return None

    async def _transcribe_clip(self, video_path: Path) -> Optional[str]:
        """Transcribe el audio de un clip usando Whisper."""
        try:
            import whisper
        except ImportError:
            return None

        try:
            # Cargar modelo tiny para velocidad
            model = whisper.load_model("tiny")

            # Transcribir
            result = await asyncio.to_thread(
                model.transcribe,
                str(video_path),
                language="es",
                fp16=False,
            )

            transcript = result.get("text", "").strip()
            if transcript:
                print(f"[Pipeline] Transcripción: {transcript[:100]}...")
            return transcript

        except Exception as e:
            print(f"[Pipeline] Transcription error: {e}")
            return None

    async def start_buffer_for_channel(self, channel: str, platform: str = "kick"):
        """Inicia el buffer para un canal específico."""
        if self.multi_buffer:
            return await self.multi_buffer.add_channel(channel, platform)
        return False

    async def stop_buffer_for_channel(self, channel: str, platform: str = "kick"):
        """Detiene el buffer para un canal específico."""
        if self.multi_buffer:
            await self.multi_buffer.remove_channel(channel, platform)

    async def capture_from_buffer(
        self,
        channel: str,
        platform: str = "kick",
        pre_seconds: int = 10,
        post_seconds: int = 20,
    ) -> Optional[Path]:
        """Captura un clip del buffer (incluye segundos ANTES del evento)."""
        if self.multi_buffer:
            return await self.multi_buffer.capture_clip(
                channel=channel,
                platform=platform,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
            )
        return None

    def get_recent_moments(self, limit: int = 20) -> List[Dict]:
        """Retorna los momentos más recientes."""
        return [
            {
                "id": m.id,
                "type": m.event_type,
                "channel": m.channel,
                "timestamp": m.timestamp.isoformat(),
                "status": m.status,
                "clip_path": str(m.clip_path) if m.clip_path else None,
                "vertical_path": str(m.vertical_path) if m.vertical_path else None,
                "processed_path": str(m.processed_path) if m.processed_path else None,
                "data": m.data,
                # AI Analysis data
                "ai_score": m.ai_score,
                "ai_title": m.ai_title,
                "ai_tags": m.ai_tags,
            }
            for m in self.moments[:limit]
        ]

    def get_stats(self) -> Dict[str, int]:
        """Retorna estadísticas del pipeline."""
        stats = {
            "total": len(self.moments),
            "detected": 0,
            "downloading": 0,
            "reframing": 0,
            "captioning": 0,
            "ready": 0,
            "published": 0,
            "failed": 0,
        }

        for m in self.moments:
            if m.status in stats:
                stats[m.status] += 1

        return stats


# ============================================
# INTEGRACIÓN CON APP
# ============================================

# Instancia global del pipeline
pipeline = ViralPipeline(
    output_dir=config.CLIPS_DIR,
    auto_download=True,
    auto_process=True,
    auto_upload=False,  # Publicación manual desde UI
    clip_duration=30,   # 30 segundos por clip
)


async def on_viral_event(event: MonitorEvent):
    """Handler global para eventos virales."""
    moment = await pipeline.handle_event(event)
    if moment:
        print(f"[Pipeline] Momento viral detectado: {moment.id}")


# Conectar a los handlers existentes
def setup_pipeline(kick_monitor):
    """Configura el pipeline con los monitores."""
    if kick_monitor:
        kick_monitor.on_event(EventType.CHAT_SPIKE, on_viral_event)

    return pipeline
