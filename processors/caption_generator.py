"""
Generador de subtítulos automáticos usando Whisper.

Transcribe el audio del video y genera subtítulos
en formato SRT/ASS con estilo TikTok/Shorts.
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import json
import re

# Whisper para transcripción
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


class CaptionStyle(Enum):
    """Estilos de subtítulos."""
    SIMPLE = "simple"          # Texto blanco simple
    TIKTOK = "tiktok"          # Estilo TikTok (palabra por palabra, coloreado)
    YOUTUBE = "youtube"        # Estilo YouTube Shorts
    GAMING = "gaming"          # Estilo gaming (neon, glow)
    MINIMAL = "minimal"        # Minimalista


@dataclass
class Caption:
    """Un segmento de subtítulo."""
    text: str
    start_time: float  # segundos
    end_time: float    # segundos
    words: List[Dict[str, Any]] = field(default_factory=list)  # Palabras con timestamps

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_srt_time(self, seconds: float) -> str:
        """Convierte segundos a formato SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_srt(self, index: int) -> str:
        """Genera bloque SRT."""
        return (
            f"{index}\n"
            f"{self.to_srt_time(self.start_time)} --> {self.to_srt_time(self.end_time)}\n"
            f"{self.text}\n"
        )


@dataclass
class CaptionResult:
    """Resultado de la generación de subtítulos."""
    video_path: Path
    srt_path: Path
    ass_path: Optional[Path] = None
    language: str = "en"
    total_captions: int = 0
    total_duration: float = 0.0
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "srt_path": str(self.srt_path),
            "ass_path": str(self.ass_path) if self.ass_path else None,
            "language": self.language,
            "total_captions": self.total_captions,
            "total_duration": self.total_duration,
            "processing_time": self.processing_time,
        }


@dataclass
class CaptionOptions:
    """Opciones de generación de subtítulos."""
    # Whisper
    model: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # None = auto-detect
    # Estilo
    style: CaptionStyle = CaptionStyle.TIKTOK
    font_name: str = "Arial Bold"
    font_size: int = 48
    primary_color: str = "FFFFFF"  # Blanco
    outline_color: str = "000000"  # Negro
    outline_width: int = 3
    # Posición
    position: str = "bottom"  # top, center, bottom
    margin_bottom: int = 100
    margin_sides: int = 50
    # Formato
    max_chars_per_line: int = 30
    max_words_per_caption: int = 8
    word_by_word: bool = True  # Mostrar palabra por palabra (estilo TikTok)


class CaptionGenerator:
    """
    Generador de subtítulos automáticos.

    Usa Whisper para transcribir y genera archivos SRT/ASS
    con estilos personalizables.

    Ejemplo:
        generator = CaptionGenerator()
        result = await generator.generate(
            Path("./video.mp4"),
            options=CaptionOptions(style=CaptionStyle.TIKTOK),
        )
        print(f"Subtítulos: {result.srt_path}")
    """

    # Plantilla ASS para estilo TikTok (ahora cuadrado 1:1)
    ASS_TEMPLATE = """[Script Info]
Title: Auto-generated Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{events}
"""

    def __init__(
        self,
        default_options: Optional[CaptionOptions] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            default_options: Opciones por defecto
            cache_dir: Directorio para cache de modelos Whisper
        """
        self.options = default_options or CaptionOptions()
        self.cache_dir = cache_dir

        self._model = None
        self._model_name = None

        # Buscar ffmpeg en el sistema
        self._ffmpeg_path = self._find_ffmpeg()

        self._stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_duration": 0.0,
        }

    def _find_ffmpeg(self) -> str:
        """Encuentra ffmpeg en el sistema."""
        # Path específico de WinGet
        winget_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        if winget_path.exists():
            # Buscar recursivamente
            for ffmpeg_exe in winget_path.rglob("ffmpeg.exe"):
                if "bin" in str(ffmpeg_exe):
                    return str(ffmpeg_exe)
        return "ffmpeg"

    def _load_model(self, model_name: str) -> None:
        """Carga el modelo de Whisper si es necesario."""
        if not HAS_WHISPER:
            raise ImportError(
                "Whisper no instalado. Ejecuta: pip install openai-whisper"
            )

        if self._model is None or self._model_name != model_name:
            self._model = whisper.load_model(
                model_name,
                download_root=str(self.cache_dir) if self.cache_dir else None,
            )
            self._model_name = model_name

    async def generate(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None,
        options: Optional[CaptionOptions] = None,
    ) -> Optional[CaptionResult]:
        """
        Genera subtítulos para un video.

        Args:
            video_path: Path del video
            output_dir: Directorio de salida (default: mismo que video)
            options: Opciones de generación

        Returns:
            CaptionResult o None si falla
        """
        opts = options or self.options

        if not video_path.exists():
            return None

        self._stats["total_processed"] += 1

        import time
        start_time = time.time()

        try:
            # Directorio de salida
            out_dir = output_dir or video_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # Extraer audio
            audio_path = await self._extract_audio(video_path)
            if not audio_path:
                self._stats["failed"] += 1
                return None

            # Transcribir con Whisper
            captions = await self._transcribe(audio_path, opts)

            # Limpiar audio temporal
            audio_path.unlink(missing_ok=True)

            if not captions:
                self._stats["failed"] += 1
                return None

            # Generar archivos
            stem = video_path.stem
            srt_path = out_dir / f"{stem}.srt"
            ass_path = out_dir / f"{stem}.ass"

            # Generar SRT
            self._write_srt(captions, srt_path)

            # Generar ASS con estilo
            self._write_ass(captions, ass_path, opts)

            # Calcular duración total
            total_duration = captions[-1].end_time if captions else 0

            processing_time = time.time() - start_time

            self._stats["successful"] += 1
            self._stats["total_duration"] += total_duration

            return CaptionResult(
                video_path=video_path,
                srt_path=srt_path,
                ass_path=ass_path,
                language=opts.language or "auto",
                total_captions=len(captions),
                total_duration=total_duration,
                processing_time=processing_time,
            )

        except Exception as e:
            self._stats["failed"] += 1
            return None

    async def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """Extrae el audio del video."""
        audio_path = video_path.with_suffix(".temp.wav")

        cmd = [
            self._ffmpeg_path,
            "-y",
            "-i", str(video_path.resolve()),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path.resolve()),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

            if process.returncode != 0:
                print(f"[CaptionGenerator] FFmpeg error: {stderr.decode('utf-8', errors='ignore')[:200]}")

            if audio_path.exists():
                return audio_path

            return None

        except Exception as e:
            print(f"[CaptionGenerator] Error extrayendo audio: {e}")
            return None

    async def _transcribe(
        self,
        audio_path: Path,
        opts: CaptionOptions,
    ) -> List[Caption]:
        """Transcribe el audio usando Whisper."""
        # Cargar modelo (esto es síncrono)
        self._load_model(opts.model)

        # Transcribir (también síncrono, ejecutar en thread pool)
        result = await asyncio.to_thread(
            self._model.transcribe,
            str(audio_path),
            language=opts.language,
            word_timestamps=True,
            verbose=False,
        )

        captions = []

        for segment in result.get("segments", []):
            # Obtener palabras con timestamps
            words = []
            if "words" in segment:
                for word_info in segment["words"]:
                    words.append({
                        "word": word_info.get("word", "").strip(),
                        "start": word_info.get("start", 0),
                        "end": word_info.get("end", 0),
                    })

            caption = Caption(
                text=segment.get("text", "").strip(),
                start_time=segment.get("start", 0),
                end_time=segment.get("end", 0),
                words=words,
            )

            captions.append(caption)

        # Post-procesar para dividir en líneas más cortas
        processed = self._split_long_captions(captions, opts)

        return processed

    def _split_long_captions(
        self,
        captions: List[Caption],
        opts: CaptionOptions,
    ) -> List[Caption]:
        """Divide subtítulos largos en líneas más cortas."""
        processed = []

        for caption in captions:
            words = caption.text.split()

            if len(words) <= opts.max_words_per_caption:
                processed.append(caption)
                continue

            # Dividir en chunks
            for i in range(0, len(words), opts.max_words_per_caption):
                chunk_words = words[i:i + opts.max_words_per_caption]
                chunk_text = " ".join(chunk_words)

                # Calcular tiempos aproximados
                word_duration = caption.duration / len(words)
                start = caption.start_time + (i * word_duration)
                end = start + (len(chunk_words) * word_duration)

                # Obtener palabras con timestamps si están disponibles
                chunk_word_info = caption.words[i:i + opts.max_words_per_caption] if caption.words else []

                new_caption = Caption(
                    text=chunk_text,
                    start_time=start,
                    end_time=end,
                    words=chunk_word_info,
                )

                processed.append(new_caption)

        return processed

    def _write_srt(self, captions: List[Caption], output_path: Path) -> None:
        """Escribe archivo SRT."""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, caption in enumerate(captions, 1):
                f.write(caption.to_srt(i))
                f.write("\n")

    def _write_ass(
        self,
        captions: List[Caption],
        output_path: Path,
        opts: CaptionOptions,
    ) -> None:
        """Escribe archivo ASS con estilos."""
        # Generar estilos según el tipo
        styles = self._generate_ass_styles(opts)

        # Generar eventos
        events = self._generate_ass_events(captions, opts)

        content = self.ASS_TEMPLATE.format(styles=styles, events=events)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _generate_ass_styles(self, opts: CaptionOptions) -> str:
        """Genera la sección de estilos ASS."""
        # Convertir colores a formato ASS (BGR con alpha)
        primary = f"&H00{opts.primary_color}&"
        outline = f"&H00{opts.outline_color}&"

        # Alignment basado en posición
        alignment = {"bottom": 2, "center": 5, "top": 8}.get(opts.position, 2)

        if opts.style == CaptionStyle.TIKTOK:
            # Estilo TikTok: texto grande, outline grueso, centrado
            return (
                f"Style: Default,{opts.font_name},{opts.font_size},"
                f"{primary},&H000000FF,{outline},&H80000000,"
                f"-1,0,0,0,100,100,0,0,1,{opts.outline_width},2,"
                f"{alignment},{opts.margin_sides},{opts.margin_sides},{opts.margin_bottom},1"
            )

        elif opts.style == CaptionStyle.GAMING:
            # Estilo gaming: neon glow
            return (
                f"Style: Default,{opts.font_name},{opts.font_size},"
                f"&H0000FFFF,&H000000FF,&H00FF00FF,&H80000000,"
                f"-1,0,0,0,100,100,0,0,1,4,3,"
                f"{alignment},{opts.margin_sides},{opts.margin_sides},{opts.margin_bottom},1"
            )

        else:
            # Estilo simple
            return (
                f"Style: Default,{opts.font_name},{opts.font_size},"
                f"{primary},&H000000FF,{outline},&H80000000,"
                f"0,0,0,0,100,100,0,0,1,{opts.outline_width},0,"
                f"{alignment},{opts.margin_sides},{opts.margin_sides},{opts.margin_bottom},1"
            )

    def _generate_ass_events(
        self,
        captions: List[Caption],
        opts: CaptionOptions,
    ) -> str:
        """Genera la sección de eventos ASS."""
        events = []

        for caption in captions:
            start = self._seconds_to_ass_time(caption.start_time)
            end = self._seconds_to_ass_time(caption.end_time)

            if opts.word_by_word and caption.words:
                # Generar efecto palabra por palabra
                text = self._generate_word_by_word(caption, opts)
            else:
                text = caption.text

            event = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            events.append(event)

        return "\n".join(events)

    def _generate_word_by_word(
        self,
        caption: Caption,
        opts: CaptionOptions,
    ) -> str:
        """Genera texto con efecto palabra por palabra."""
        # Esto crea un efecto donde cada palabra aparece progresivamente
        # Usando tags ASS para karaoke

        if not caption.words:
            return caption.text

        parts = []
        for word_info in caption.words:
            word = word_info.get("word", "")
            word_start = word_info.get("start", caption.start_time)
            word_end = word_info.get("end", caption.end_time)

            # Calcular duración en centisegundos
            duration_cs = int((word_end - word_start) * 100)

            # Tag de karaoke
            parts.append(f"{{\\k{duration_cs}}}{word}")

        return "".join(parts)

    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convierte segundos a formato ASS (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

    async def burn_captions(
        self,
        video_path: Path,
        caption_path: Path,
        output_path: Path,
        options: Optional[CaptionOptions] = None,
    ) -> bool:
        """
        Incrusta los subtítulos en el video.

        Args:
            video_path: Video original
            caption_path: Archivo de subtítulos (SRT o ASS)
            output_path: Video de salida

        Returns:
            True si fue exitoso
        """
        opts = options or self.options

        # En Windows, FFmpeg tiene problemas con rutas que contienen espacios
        # Necesitamos escapar correctamente para el filtro
        # Convertir a ruta absoluta y escapar barras invertidas y dos puntos
        caption_str = str(caption_path.absolute())
        # FFmpeg en Windows necesita: reemplazar \ por / o \\, y escapar : con \\:
        caption_escaped = caption_str.replace("\\", "/").replace(":", "\\:")

        # Determinar filtro según tipo de archivo
        if caption_path.suffix.lower() == ".ass":
            subtitle_filter = f"ass='{caption_escaped}'"
        else:
            subtitle_filter = f"subtitles='{caption_escaped}':force_style='FontSize={opts.font_size}'"

        cmd = [
            self._ffmpeg_path,
            "-y",
            "-i", str(video_path.resolve()),
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_path.resolve()),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            else:
                # Log error para debug
                print(f"[CaptionGenerator] FFmpeg error: {stderr.decode('utf-8', errors='ignore')[:500]}")
                return False

        except Exception as e:
            print(f"[CaptionGenerator] Exception: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de procesamiento."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful"] / self._stats["total_processed"]
                if self._stats["total_processed"] > 0 else 0
            ),
        }


# =====================================================
# FUNCIONES HELPER
# =====================================================

async def quick_caption(
    video_path: Path,
    style: CaptionStyle = CaptionStyle.TIKTOK,
    burn: bool = False,
) -> Optional[Path]:
    """
    Función rápida para generar subtítulos.

    Args:
        video_path: Path del video
        style: Estilo de subtítulos
        burn: Si incrustar subtítulos en el video

    Returns:
        Path del archivo de subtítulos o video con subtítulos
    """
    generator = CaptionGenerator(
        default_options=CaptionOptions(style=style)
    )

    result = await generator.generate(video_path)

    if not result:
        return None

    if burn:
        output_path = video_path.with_name(f"{video_path.stem}_captioned.mp4")
        success = await generator.burn_captions(
            video_path,
            result.ass_path,
            output_path,
        )
        return output_path if success else None

    return result.srt_path


# =====================================================
# EJEMPLO DE USO
# =====================================================

async def main():
    """Ejemplo de uso del CaptionGenerator."""
    from core.logger import logger

    if not HAS_WHISPER:
        logger.error("Whisper no instalado. Ejecuta: pip install openai-whisper")
        return

    generator = CaptionGenerator(
        default_options=CaptionOptions(
            model="base",  # tiny es más rápido, large es más preciso
            style=CaptionStyle.TIKTOK,
            font_size=48,
            word_by_word=True,
        )
    )

    # Video de prueba (reemplazar con path real)
    video_path = Path("./clips/test_video.mp4")

    if not video_path.exists():
        logger.warning(f"Video de prueba no encontrado: {video_path}")
        return

    logger.info(f"Generando subtítulos para: {video_path}")

    result = await generator.generate(video_path)

    if result:
        logger.info(f"SRT: {result.srt_path}")
        logger.info(f"ASS: {result.ass_path}")
        logger.info(f"Subtítulos: {result.total_captions}")
        logger.info(f"Tiempo: {result.processing_time:.2f}s")

        # Opcional: incrustar subtítulos
        # output = video_path.with_name(f"{video_path.stem}_captioned.mp4")
        # await generator.burn_captions(video_path, result.ass_path, output)

    logger.info(f"Stats: {generator.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
