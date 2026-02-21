"""
Compilador de videos para crear compilaciones de highlights.

Une múltiples clips en un solo video con:
- Transiciones
- Música de fondo (opcional)
- Intro/Outro
- Marcas de agua
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# MoviePy para edición de video
try:
    from moviepy.editor import (
        VideoFileClip,
        AudioFileClip,
        CompositeVideoClip,
        CompositeAudioClip,
        concatenate_videoclips,
        TextClip,
        ImageClip,
    )
    from moviepy.video.fx.all import fadein, fadeout, crossfadein, crossfadeout
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout, volumex
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    # Crear clases dummy para type hints
    VideoFileClip = None
    AudioFileClip = None
    CompositeVideoClip = None


class TransitionType(Enum):
    """Tipos de transición entre clips."""
    NONE = "none"
    FADE = "fade"
    CROSSFADE = "crossfade"
    WIPE = "wipe"
    SLIDE = "slide"


@dataclass
class ClipConfig:
    """Configuración de un clip individual."""
    path: Path
    start_time: Optional[float] = None  # Recortar desde
    end_time: Optional[float] = None    # Recortar hasta
    volume: float = 1.0                  # Volumen (0-1)
    speed: float = 1.0                   # Velocidad (1.0 = normal)
    label: Optional[str] = None          # Texto overlay


@dataclass
class CompilationResult:
    """Resultado de una compilación."""
    output_path: Path
    total_clips: int
    total_duration: float
    resolution: Tuple[int, int]
    processing_time: float
    clips_used: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "total_clips": self.total_clips,
            "total_duration": self.total_duration,
            "resolution": self.resolution,
            "processing_time": self.processing_time,
            "clips_used": self.clips_used,
        }


@dataclass
class CompilationOptions:
    """Opciones de compilación."""
    # Video
    output_width: int = 1080
    output_height: int = 1920  # Vertical por defecto
    fps: int = 30
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    bitrate: str = "8M"
    # Transiciones
    transition_type: TransitionType = TransitionType.CROSSFADE
    transition_duration: float = 0.5  # segundos
    # Audio
    background_music_path: Optional[Path] = None
    background_music_volume: float = 0.3
    # Intro/Outro
    intro_clip_path: Optional[Path] = None
    outro_clip_path: Optional[Path] = None
    # Watermark
    watermark_path: Optional[Path] = None
    watermark_position: str = "bottom-right"  # top-left, top-right, bottom-left, bottom-right
    watermark_opacity: float = 0.7
    watermark_scale: float = 0.15
    # Texto
    show_clip_labels: bool = False
    label_font: str = "Arial-Bold"
    label_size: int = 40


class VideoCompiler:
    """
    Compilador de videos para crear compilaciones.

    Une múltiples clips con transiciones, música de fondo,
    y elementos adicionales como watermarks y texto.

    Ejemplo:
        compiler = VideoCompiler(output_dir=Path("./compilations"))
        clips = [
            ClipConfig(Path("clip1.mp4")),
            ClipConfig(Path("clip2.mp4"), label="Epic Win!"),
            ClipConfig(Path("clip3.mp4")),
        ]
        result = await compiler.compile(clips, "my_compilation.mp4")
        print(f"Compilación: {result.output_path}")
    """

    def __init__(
        self,
        output_dir: Path,
        default_options: Optional[CompilationOptions] = None,
    ):
        """
        Args:
            output_dir: Directorio de salida
            default_options: Opciones por defecto
        """
        if not HAS_MOVIEPY:
            raise ImportError(
                "MoviePy no instalado. Ejecuta: pip install moviepy"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.options = default_options or CompilationOptions()

        self._stats = {
            "total_compilations": 0,
            "successful": 0,
            "failed": 0,
            "total_clips_processed": 0,
        }

    async def compile(
        self,
        clips: List[Union[ClipConfig, Path]],
        output_filename: str,
        options: Optional[CompilationOptions] = None,
    ) -> Optional[CompilationResult]:
        """
        Compila múltiples clips en un solo video.

        Args:
            clips: Lista de clips (ClipConfig o Path)
            output_filename: Nombre del archivo de salida
            options: Opciones de compilación

        Returns:
            CompilationResult o None si falla
        """
        opts = options or self.options

        if not clips:
            return None

        self._stats["total_compilations"] += 1

        import time
        start_time = time.time()

        try:
            # Normalizar clips a ClipConfig
            clip_configs = []
            for clip in clips:
                if isinstance(clip, Path):
                    clip_configs.append(ClipConfig(path=clip))
                else:
                    clip_configs.append(clip)

            # Verificar que existen
            valid_clips = [c for c in clip_configs if c.path.exists()]
            if not valid_clips:
                self._stats["failed"] += 1
                return None

            # Cargar clips
            loaded_clips = await self._load_clips(valid_clips, opts)
            if not loaded_clips:
                self._stats["failed"] += 1
                return None

            # Aplicar transiciones
            if opts.transition_type != TransitionType.NONE and len(loaded_clips) > 1:
                final_clip = self._apply_transitions(loaded_clips, opts)
            else:
                final_clip = concatenate_videoclips(loaded_clips, method="compose")

            # Añadir intro/outro
            if opts.intro_clip_path and opts.intro_clip_path.exists():
                intro = await self._load_single_clip(opts.intro_clip_path, opts)
                if intro:
                    final_clip = concatenate_videoclips([intro, final_clip])

            if opts.outro_clip_path and opts.outro_clip_path.exists():
                outro = await self._load_single_clip(opts.outro_clip_path, opts)
                if outro:
                    final_clip = concatenate_videoclips([final_clip, outro])

            # Añadir watermark
            if opts.watermark_path and opts.watermark_path.exists():
                final_clip = self._add_watermark(final_clip, opts)

            # Añadir música de fondo
            if opts.background_music_path and opts.background_music_path.exists():
                final_clip = await self._add_background_music(final_clip, opts)

            # Exportar
            output_path = self.output_dir / output_filename

            # Ejecutar write_videofile en thread pool
            await asyncio.to_thread(
                self._export_video,
                final_clip,
                output_path,
                opts,
            )

            # Limpiar
            for clip in loaded_clips:
                clip.close()
            final_clip.close()

            processing_time = time.time() - start_time

            self._stats["successful"] += 1
            self._stats["total_clips_processed"] += len(valid_clips)

            return CompilationResult(
                output_path=output_path,
                total_clips=len(valid_clips),
                total_duration=final_clip.duration if hasattr(final_clip, 'duration') else 0,
                resolution=(opts.output_width, opts.output_height),
                processing_time=processing_time,
                clips_used=[str(c.path) for c in valid_clips],
            )

        except Exception as e:
            self._stats["failed"] += 1
            return None

    async def _load_clips(
        self,
        configs: List[ClipConfig],
        opts: CompilationOptions,
    ) -> List[VideoFileClip]:
        """Carga y preprocesa los clips."""
        loaded = []

        for config in configs:
            clip = await self._load_single_clip(config.path, opts)
            if not clip:
                continue

            # Recortar si es necesario
            if config.start_time is not None or config.end_time is not None:
                start = config.start_time or 0
                end = config.end_time or clip.duration
                clip = clip.subclip(start, end)

            # Ajustar velocidad
            if config.speed != 1.0:
                clip = clip.speedx(config.speed)

            # Ajustar volumen
            if config.volume != 1.0 and clip.audio:
                clip = clip.volumex(config.volume)

            # Añadir label si está configurado
            if config.label and opts.show_clip_labels:
                clip = self._add_label(clip, config.label, opts)

            loaded.append(clip)

        return loaded

    async def _load_single_clip(
        self,
        path: Path,
        opts: CompilationOptions,
    ) -> Optional[VideoFileClip]:
        """Carga un solo clip y lo redimensiona."""
        try:
            clip = VideoFileClip(str(path))

            # Redimensionar al tamaño de salida
            clip = clip.resize((opts.output_width, opts.output_height))

            return clip

        except Exception:
            return None

    def _apply_transitions(
        self,
        clips: List[VideoFileClip],
        opts: CompilationOptions,
    ) -> VideoFileClip:
        """Aplica transiciones entre clips."""
        if opts.transition_type == TransitionType.FADE:
            # Fade in/out en cada clip
            processed = []
            for i, clip in enumerate(clips):
                if i > 0:
                    clip = fadein(clip, opts.transition_duration)
                if i < len(clips) - 1:
                    clip = fadeout(clip, opts.transition_duration)
                processed.append(clip)

            return concatenate_videoclips(processed, method="compose")

        elif opts.transition_type == TransitionType.CROSSFADE:
            # Crossfade entre clips
            # MoviePy no tiene crossfade nativo fácil, usar concatenate con padding
            return concatenate_videoclips(
                clips,
                padding=-opts.transition_duration,
                method="compose",
            )

        else:
            return concatenate_videoclips(clips, method="compose")

    def _add_watermark(
        self,
        clip: VideoFileClip,
        opts: CompilationOptions,
    ) -> CompositeVideoClip:
        """Añade un watermark al video."""
        try:
            watermark = ImageClip(str(opts.watermark_path))

            # Escalar
            wm_width = int(opts.output_width * opts.watermark_scale)
            watermark = watermark.resize(width=wm_width)

            # Opacidad
            watermark = watermark.set_opacity(opts.watermark_opacity)

            # Posición
            positions = {
                "top-left": ("left", "top"),
                "top-right": ("right", "top"),
                "bottom-left": ("left", "bottom"),
                "bottom-right": ("right", "bottom"),
            }
            pos = positions.get(opts.watermark_position, ("right", "bottom"))

            # Margen
            margin = 20
            if pos[0] == "left":
                x = margin
            else:
                x = opts.output_width - watermark.w - margin

            if pos[1] == "top":
                y = margin
            else:
                y = opts.output_height - watermark.h - margin

            watermark = watermark.set_position((x, y))
            watermark = watermark.set_duration(clip.duration)

            return CompositeVideoClip([clip, watermark])

        except Exception:
            return clip

    def _add_label(
        self,
        clip: VideoFileClip,
        text: str,
        opts: CompilationOptions,
    ) -> CompositeVideoClip:
        """Añade un texto label al clip."""
        try:
            label = TextClip(
                text,
                fontsize=opts.label_size,
                font=opts.label_font,
                color="white",
                stroke_color="black",
                stroke_width=2,
            )

            # Posición: arriba centrado
            label = label.set_position(("center", 50))
            label = label.set_duration(min(3.0, clip.duration))  # Mostrar 3 segundos max

            return CompositeVideoClip([clip, label])

        except Exception:
            return clip

    async def _add_background_music(
        self,
        clip: VideoFileClip,
        opts: CompilationOptions,
    ) -> VideoFileClip:
        """Añade música de fondo al video."""
        try:
            music = AudioFileClip(str(opts.background_music_path))

            # Ajustar duración
            if music.duration < clip.duration:
                # Loop la música
                loops_needed = int(clip.duration / music.duration) + 1
                music = concatenate_audioclips([music] * loops_needed)

            music = music.subclip(0, clip.duration)

            # Ajustar volumen
            music = music.volumex(opts.background_music_volume)

            # Combinar con audio original
            if clip.audio:
                final_audio = CompositeAudioClip([clip.audio, music])
            else:
                final_audio = music

            return clip.set_audio(final_audio)

        except Exception:
            return clip

    def _export_video(
        self,
        clip: VideoFileClip,
        output_path: Path,
        opts: CompilationOptions,
    ) -> None:
        """Exporta el video (síncrono, llamar desde thread pool)."""
        clip.write_videofile(
            str(output_path),
            fps=opts.fps,
            codec=opts.video_codec,
            audio_codec=opts.audio_codec,
            bitrate=opts.bitrate,
            preset="medium",
            verbose=False,
            logger=None,
        )

    async def create_highlight_reel(
        self,
        clips_dir: Path,
        output_filename: str,
        max_clips: int = 10,
        max_duration_per_clip: float = 10.0,
        options: Optional[CompilationOptions] = None,
    ) -> Optional[CompilationResult]:
        """
        Crea un highlight reel automático desde un directorio de clips.

        Args:
            clips_dir: Directorio con clips
            output_filename: Nombre del archivo de salida
            max_clips: Máximo de clips a incluir
            max_duration_per_clip: Duración máxima por clip

        Returns:
            CompilationResult o None
        """
        if not clips_dir.exists():
            return None

        # Buscar clips de video
        video_extensions = [".mp4", ".webm", ".mkv", ".mov", ".avi"]
        clip_files = []

        for ext in video_extensions:
            clip_files.extend(clips_dir.glob(f"*{ext}"))

        if not clip_files:
            return None

        # Ordenar por fecha de modificación (más recientes primero)
        clip_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Limitar cantidad
        clip_files = clip_files[:max_clips]

        # Crear configs con duración limitada
        configs = []
        for clip_path in clip_files:
            config = ClipConfig(
                path=clip_path,
                end_time=max_duration_per_clip,
            )
            configs.append(config)

        return await self.compile(configs, output_filename, options)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de compilaciones."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful"] / self._stats["total_compilations"]
                if self._stats["total_compilations"] > 0 else 0
            ),
        }


# =====================================================
# FUNCIONES HELPER
# =====================================================

async def quick_compile(
    clip_paths: List[Path],
    output_path: Path,
    vertical: bool = True,
) -> Optional[Path]:
    """
    Función rápida para compilar clips.

    Args:
        clip_paths: Lista de paths de clips
        output_path: Path del video de salida
        vertical: Si usar formato vertical (9:16)

    Returns:
        Path del video compilado o None
    """
    output_dir = output_path.parent
    output_name = output_path.name

    opts = CompilationOptions(
        output_width=1080 if vertical else 1920,
        output_height=1920 if vertical else 1080,
        transition_type=TransitionType.CROSSFADE,
    )

    compiler = VideoCompiler(output_dir=output_dir, default_options=opts)

    configs = [ClipConfig(path=p) for p in clip_paths]

    result = await compiler.compile(configs, output_name)

    return result.output_path if result else None


# =====================================================
# EJEMPLO DE USO
# =====================================================

async def main():
    """Ejemplo de uso del VideoCompiler."""
    from core.logger import logger

    if not HAS_MOVIEPY:
        logger.error("MoviePy no instalado. Ejecuta: pip install moviepy")
        return

    output_dir = Path("./compilations")
    compiler = VideoCompiler(
        output_dir=output_dir,
        default_options=CompilationOptions(
            transition_type=TransitionType.CROSSFADE,
            transition_duration=0.5,
            show_clip_labels=True,
        ),
    )

    # Clips de prueba (reemplazar con paths reales)
    clips_dir = Path("./clips")

    if not clips_dir.exists():
        logger.warning(f"Directorio de clips no encontrado: {clips_dir}")
        logger.info("Crea algunos clips de prueba primero")
        return

    # Crear highlight reel automático
    result = await compiler.create_highlight_reel(
        clips_dir=clips_dir,
        output_filename="highlights_auto.mp4",
        max_clips=5,
        max_duration_per_clip=8.0,
    )

    if result:
        logger.info(f"Compilación creada: {result.output_path}")
        logger.info(f"Clips: {result.total_clips}")
        logger.info(f"Duración: {result.total_duration:.2f}s")
        logger.info(f"Tiempo de procesamiento: {result.processing_time:.2f}s")
    else:
        logger.error("Error al crear compilación")

    logger.info(f"Stats: {compiler.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
