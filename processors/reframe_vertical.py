"""
Reframe de video horizontal (16:9) a vertical (9:16).

Usa detección de rostros/objetos para mantener el sujeto
principal centrado en el nuevo encuadre.
"""

import asyncio
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import json

# MoviePy para edición de video
try:
    from moviepy.editor import VideoFileClip, CompositeVideoClip
    from moviepy.video.fx.all import crop, resize
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

# OpenCV para detección de rostros
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class ReframeMode(Enum):
    """Modos de reencuadre."""
    CENTER = "center"              # Centrado simple
    FACE_TRACKING = "face"         # Seguir rostros
    OBJECT_TRACKING = "object"     # Seguir objetos
    SMART = "smart"                # Combinar métodos
    MANUAL = "manual"              # Posición manual


@dataclass
class ReframeResult:
    """Resultado del reencuadre."""
    input_path: Path
    output_path: Path
    original_resolution: Tuple[int, int]
    output_resolution: Tuple[int, int]
    duration: float
    mode_used: ReframeMode
    faces_detected: int = 0
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "original_resolution": self.original_resolution,
            "output_resolution": self.output_resolution,
            "duration": self.duration,
            "mode_used": self.mode_used.value,
            "faces_detected": self.faces_detected,
            "processing_time": self.processing_time,
        }


@dataclass
class ReframeOptions:
    """Opciones de reencuadre."""
    output_width: int = 1080
    output_height: int = 1920
    mode: ReframeMode = ReframeMode.SMART
    fps: Optional[int] = None  # None = mantener original
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 23
    preset: str = "medium"
    # Para detección de rostros
    face_detection_interval: int = 10  # Cada N frames
    smoothing_factor: float = 0.3      # Suavizado del movimiento (0-1)
    # Padding alrededor del sujeto
    padding_percent: float = 0.1       # 10% de padding


class FaceTracker:
    """
    Tracker de rostros para reencuadre inteligente.

    Usa OpenCV para detectar rostros y calcular
    la posición óptima del crop.
    """

    def __init__(self):
        if not HAS_OPENCV:
            raise ImportError("OpenCV no instalado. Ejecuta: pip install opencv-python")

        # Cargar clasificador de rostros
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Estado del tracking
        self._last_position: Optional[Tuple[int, int]] = None
        self._smoothing = 0.3

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostros en un frame.

        Args:
            frame: Frame en formato BGR (OpenCV)

        Returns:
            Lista de (x, y, w, h) para cada rostro
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )

        return [(x, y, w, h) for (x, y, w, h) in faces]

    def get_focus_point(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int]:
        """
        Calcula el punto focal basado en los rostros detectados.

        Args:
            frame: Frame en formato BGR
            frame_width: Ancho del frame
            frame_height: Alto del frame

        Returns:
            (x, y) del punto focal
        """
        faces = self.detect_faces(frame)

        if not faces:
            # Sin rostros, usar centro o última posición conocida
            if self._last_position:
                return self._last_position
            return (frame_width // 2, frame_height // 2)

        # Calcular centro de todos los rostros
        total_x = 0
        total_y = 0

        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            total_x += center_x
            total_y += center_y

        avg_x = total_x // len(faces)
        avg_y = total_y // len(faces)

        # Aplicar suavizado
        if self._last_position:
            smooth_x = int(
                self._last_position[0] * (1 - self._smoothing) +
                avg_x * self._smoothing
            )
            smooth_y = int(
                self._last_position[1] * (1 - self._smoothing) +
                avg_y * self._smoothing
            )
            result = (smooth_x, smooth_y)
        else:
            result = (avg_x, avg_y)

        self._last_position = result
        return result

    def reset(self) -> None:
        """Reinicia el estado del tracker."""
        self._last_position = None


class VideoReframer:
    """
    Reencuadrador de video para formato vertical.

    Convierte videos 16:9 a 9:16 manteniendo el contenido
    importante centrado.

    Ejemplo:
        reframer = VideoReframer(output_dir=Path("./vertical"))
        result = await reframer.reframe(
            Path("./input.mp4"),
            mode=ReframeMode.FACE_TRACKING,
        )
        print(f"Video vertical: {result.output_path}")
    """

    def __init__(
        self,
        output_dir: Path,
        default_options: Optional[ReframeOptions] = None,
    ):
        """
        Args:
            output_dir: Directorio de salida
            default_options: Opciones por defecto
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.options = default_options or ReframeOptions()

        self._face_tracker: Optional[FaceTracker] = None
        if HAS_OPENCV:
            self._face_tracker = FaceTracker()

        # Buscar ffmpeg en el sistema
        self._ffmpeg_path = self._find_ffmpeg()
        self._ffprobe_path = self._find_ffprobe()

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

    def _find_ffprobe(self) -> str:
        """Encuentra ffprobe en el sistema."""
        # Path específico de WinGet
        winget_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        if winget_path.exists():
            # Buscar recursivamente
            for ffprobe_exe in winget_path.rglob("ffprobe.exe"):
                if "bin" in str(ffprobe_exe):
                    return str(ffprobe_exe)
        return "ffprobe"

    async def reframe(
        self,
        input_path: Path,
        output_filename: Optional[str] = None,
        options: Optional[ReframeOptions] = None,
    ) -> Optional[ReframeResult]:
        """
        Reencuadra un video a formato vertical.

        Args:
            input_path: Path del video de entrada
            output_filename: Nombre del archivo de salida (opcional)
            options: Opciones de reencuadre

        Returns:
            ReframeResult o None si falla
        """
        opts = options or self.options

        if not input_path.exists():
            return None

        self._stats["total_processed"] += 1

        import time
        start_time = time.time()

        try:
            # Determinar nombre de salida
            if output_filename:
                output_path = self.output_dir / output_filename
            else:
                stem = input_path.stem
                output_path = self.output_dir / f"{stem}_vertical.mp4"

            # Seleccionar método de procesamiento
            if opts.mode == ReframeMode.CENTER:
                result = await self._reframe_center(input_path, output_path, opts)
            elif opts.mode == ReframeMode.FACE_TRACKING:
                result = await self._reframe_face_tracking(input_path, output_path, opts)
            elif opts.mode == ReframeMode.SMART:
                # Intentar face tracking, fallback a center
                if self._face_tracker:
                    result = await self._reframe_face_tracking(input_path, output_path, opts)
                else:
                    result = await self._reframe_center(input_path, output_path, opts)
            else:
                result = await self._reframe_center(input_path, output_path, opts)

            if result:
                result.processing_time = time.time() - start_time
                self._stats["successful"] += 1
                self._stats["total_duration"] += result.duration

            return result

        except Exception as e:
            self._stats["failed"] += 1
            return None

    async def _reframe_center(
        self,
        input_path: Path,
        output_path: Path,
        opts: ReframeOptions,
    ) -> Optional[ReframeResult]:
        """Reencuadre simple centrado usando FFmpeg."""
        # Obtener info del video
        info = await self._get_video_info(input_path)
        if not info:
            return None

        in_width = info["width"]
        in_height = info["height"]
        duration = info["duration"]

        # Calcular dimensiones del crop
        # Para 9:16, necesitamos recortar horizontalmente
        target_ratio = opts.output_width / opts.output_height  # 0.5625 para 9:16

        if in_width / in_height > target_ratio:
            # Video más ancho, recortar lados
            crop_height = in_height
            crop_width = int(in_height * target_ratio)
        else:
            # Video más alto, recortar arriba/abajo
            crop_width = in_width
            crop_height = int(in_width / target_ratio)

        # Posición del crop (centrado)
        x_offset = (in_width - crop_width) // 2
        y_offset = (in_height - crop_height) // 2

        # Construir comando FFmpeg con path absoluto
        ffmpeg_cmd = [
            self._ffmpeg_path,
            "-y",
            "-i", str(input_path.resolve()),
            "-vf", (
                f"crop={crop_width}:{crop_height}:{x_offset}:{y_offset},"
                f"scale={opts.output_width}:{opts.output_height}"
            ),
            "-c:v", opts.video_codec,
            "-crf", str(opts.crf),
            "-preset", opts.preset,
            "-c:a", opts.audio_codec,
            str(output_path.resolve()),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode != 0:
                print(f"[VideoReframer] FFmpeg error: {stderr.decode('utf-8', errors='ignore')[:200]}")
        except Exception as e:
            print(f"[VideoReframer] Error ejecutando ffmpeg: {e}")

        if output_path.exists():
            return ReframeResult(
                input_path=input_path,
                output_path=output_path,
                original_resolution=(in_width, in_height),
                output_resolution=(opts.output_width, opts.output_height),
                duration=duration,
                mode_used=ReframeMode.CENTER,
            )

        return None

    async def _reframe_face_tracking(
        self,
        input_path: Path,
        output_path: Path,
        opts: ReframeOptions,
    ) -> Optional[ReframeResult]:
        """Reencuadre con seguimiento de rostros usando MoviePy + OpenCV."""
        if not HAS_MOVIEPY or not HAS_OPENCV:
            # Fallback a centrado
            return await self._reframe_center(input_path, output_path, opts)

        try:
            # Cargar video
            clip = VideoFileClip(str(input_path))

            in_width = clip.w
            in_height = clip.h
            duration = clip.duration

            # Calcular dimensiones del crop
            target_ratio = opts.output_width / opts.output_height
            if in_width / in_height > target_ratio:
                crop_height = in_height
                crop_width = int(in_height * target_ratio)
            else:
                crop_width = in_width
                crop_height = int(in_width / target_ratio)

            # Analizar frames para detectar rostros
            focus_points = await self._analyze_video_faces(
                clip,
                crop_width,
                crop_height,
                opts.face_detection_interval,
            )

            # Crear función de crop dinámico
            def get_crop_position(t):
                frame_idx = int(t * clip.fps)
                if frame_idx < len(focus_points):
                    x, y = focus_points[frame_idx]
                else:
                    x, y = focus_points[-1] if focus_points else (in_width // 2, in_height // 2)

                # Calcular posición del crop
                x1 = max(0, min(x - crop_width // 2, in_width - crop_width))
                y1 = max(0, min(y - crop_height // 2, in_height - crop_height))

                return x1, y1

            # Aplicar crop frame por frame
            def make_frame(t):
                frame = clip.get_frame(t)
                x1, y1 = get_crop_position(t)
                cropped = frame[y1:y1+crop_height, x1:x1+crop_width]
                return cropped

            # Crear clip procesado
            from moviepy.editor import VideoClip
            processed = VideoClip(make_frame, duration=duration)
            processed = processed.set_fps(clip.fps)

            # Añadir audio
            if clip.audio:
                processed = processed.set_audio(clip.audio)

            # Resize a resolución final
            processed = processed.resize((opts.output_width, opts.output_height))

            # Exportar
            processed.write_videofile(
                str(output_path),
                codec=opts.video_codec,
                audio_codec=opts.audio_codec,
                preset=opts.preset,
                ffmpeg_params=["-crf", str(opts.crf)],
                verbose=False,
                logger=None,
            )

            clip.close()

            faces_count = sum(1 for p in focus_points if p != (in_width // 2, in_height // 2))

            return ReframeResult(
                input_path=input_path,
                output_path=output_path,
                original_resolution=(in_width, in_height),
                output_resolution=(opts.output_width, opts.output_height),
                duration=duration,
                mode_used=ReframeMode.FACE_TRACKING,
                faces_detected=faces_count,
            )

        except Exception as e:
            # Fallback a centrado
            return await self._reframe_center(input_path, output_path, opts)

    async def _analyze_video_faces(
        self,
        clip,
        crop_width: int,
        crop_height: int,
        interval: int,
    ) -> List[Tuple[int, int]]:
        """Analiza el video para detectar posiciones de rostros."""
        focus_points = []
        total_frames = int(clip.duration * clip.fps)

        self._face_tracker.reset()

        for i in range(total_frames):
            if i % interval == 0:
                # Analizar este frame
                t = i / clip.fps
                frame = clip.get_frame(t)

                # Convertir RGB a BGR para OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                point = self._face_tracker.get_focus_point(
                    frame_bgr,
                    clip.w,
                    clip.h,
                )
                focus_points.append(point)
            else:
                # Interpolar desde el último punto conocido
                if focus_points:
                    focus_points.append(focus_points[-1])
                else:
                    focus_points.append((clip.w // 2, clip.h // 2))

        return focus_points

    async def _get_video_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """Obtiene información del video usando FFprobe."""
        cmd = [
            self._ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path.resolve()),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

            if process.returncode == 0:
                data = json.loads(stdout.decode())

                # Buscar stream de video
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        # Calcular FPS de forma segura
                        fps_str = stream.get("r_frame_rate", "30/1")
                        try:
                            if "/" in fps_str:
                                num, den = fps_str.split("/")
                                fps = float(num) / float(den) if float(den) != 0 else 30
                            else:
                                fps = float(fps_str)
                        except:
                            fps = 30

                        return {
                            "width": stream.get("width", 0),
                            "height": stream.get("height", 0),
                            "duration": float(data.get("format", {}).get("duration", 0)),
                            "fps": fps,
                        }
            else:
                print(f"[VideoReframer] FFprobe error: {stderr.decode('utf-8', errors='ignore')[:100]}")

            return None

        except Exception as e:
            print(f"[VideoReframer] Error obteniendo info: {e}")
            return None

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
# FUNCIÓN RÁPIDA DE REFRAME
# =====================================================

async def quick_reframe(
    input_path: Path,
    output_dir: Path,
    mode: ReframeMode = ReframeMode.CENTER,
) -> Optional[Path]:
    """
    Función rápida para reencuadrar un video.

    Args:
        input_path: Video de entrada
        output_dir: Directorio de salida
        mode: Modo de reencuadre

    Returns:
        Path del video procesado o None
    """
    reframer = VideoReframer(output_dir=output_dir)
    result = await reframer.reframe(
        input_path,
        options=ReframeOptions(mode=mode),
    )

    return result.output_path if result else None


# =====================================================
# EJEMPLO DE USO
# =====================================================

async def main():
    """Ejemplo de uso del VideoReframer."""
    from core.logger import logger

    output_dir = Path("./vertical")
    reframer = VideoReframer(
        output_dir=output_dir,
        default_options=ReframeOptions(
            mode=ReframeMode.SMART,
            output_width=1080,
            output_height=1920,
        ),
    )

    # Video de prueba (reemplazar con path real)
    input_path = Path("./clips/test_video.mp4")

    if not input_path.exists():
        logger.warning(f"Video de prueba no encontrado: {input_path}")
        logger.info("Crea un video de prueba o modifica el path")
        return

    logger.info(f"Procesando: {input_path}")

    result = await reframer.reframe(input_path)

    if result:
        logger.info(f"Video vertical creado: {result.output_path}")
        logger.info(f"Resolución: {result.original_resolution} → {result.output_resolution}")
        logger.info(f"Tiempo de procesamiento: {result.processing_time:.2f}s")
    else:
        logger.error("Error al procesar video")

    logger.info(f"Stats: {reframer.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
