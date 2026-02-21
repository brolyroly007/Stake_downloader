"""
Renderizador de replays de Stake.

Los replays de Stake NO son archivos de video descargables.
Son datos (semilla, nonce, etc.) que se renderizan en el navegador.

Este módulo usa Playwright + FFmpeg para:
1. Abrir el replay en un navegador headless
2. Capturar la pantalla mientras se reproduce
3. Guardar como archivo de video
"""

import asyncio
import subprocess
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import tempfile

# Playwright para control del navegador
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    # Tipos dummy para que el código compile sin Playwright
    Browser = None
    Page = None
    BrowserContext = None


class StakeGame(Enum):
    """Juegos de Stake soportados."""
    CRASH = "crash"
    DICE = "dice"
    MINES = "mines"
    PLINKO = "plinko"
    LIMBO = "limbo"
    KENO = "keno"
    SLOTS = "slots"  # Slots de terceros
    UNKNOWN = "unknown"


@dataclass
class ReplayInfo:
    """Información de un replay capturado."""
    replay_id: str
    game: StakeGame
    url: str
    file_path: Path
    duration: float  # segundos
    resolution: Tuple[int, int]  # (width, height)
    metadata: Dict[str, Any] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "game": self.game.value,
            "url": self.url,
            "file_path": str(self.file_path),
            "duration": self.duration,
            "resolution": self.resolution,
            "captured_at": self.captured_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CaptureOptions:
    """Opciones de captura."""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    video_codec: str = "libx264"
    audio_codec: str = "aac"  # Si hay audio
    crf: int = 23  # Calidad (menor = mejor, 18-28 recomendado)
    preset: str = "fast"  # ultrafast, fast, medium, slow
    max_duration: float = 60.0  # Máxima duración en segundos
    wait_for_load: float = 5.0  # Esperar antes de grabar
    headless: bool = False  # False para ver el navegador (debug)


class ReplayRenderer:
    """
    Renderizador de replays de Stake.

    Captura la pantalla del navegador mientras reproduce el replay
    y lo guarda como archivo de video.

    NOTA: Requiere:
    - Playwright instalado: pip install playwright && playwright install chromium
    - FFmpeg instalado y en PATH

    Ejemplo:
        renderer = ReplayRenderer(output_dir=Path("./replays"))
        replay = await renderer.capture(
            "https://stake.com/casino/games/crash?game=...&modal=replay",
            duration=30.0,
        )
        print(f"Capturado: {replay.file_path}")
    """

    # Selectores para detectar cuando el juego está listo
    GAME_READY_SELECTORS = {
        StakeGame.CRASH: ".crash-game-container",
        StakeGame.DICE: ".dice-game-container",
        StakeGame.MINES: ".mines-game-container",
        StakeGame.PLINKO: ".plinko-game-container",
    }

    def __init__(
        self,
        output_dir: Path,
        default_options: Optional[CaptureOptions] = None,
    ):
        """
        Args:
            output_dir: Directorio para guardar capturas
            default_options: Opciones por defecto
        """
        if not HAS_PLAYWRIGHT:
            raise ImportError(
                "Playwright no está instalado. "
                "Ejecuta: pip install playwright && playwright install chromium"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.options = default_options or CaptureOptions()

        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

        self._stats = {
            "total_captures": 0,
            "successful": 0,
            "failed": 0,
            "total_duration": 0.0,
        }

    def _detect_game(self, url: str) -> StakeGame:
        """Detecta el juego basándose en la URL."""
        url_lower = url.lower()

        if "/crash" in url_lower:
            return StakeGame.CRASH
        elif "/dice" in url_lower:
            return StakeGame.DICE
        elif "/mines" in url_lower:
            return StakeGame.MINES
        elif "/plinko" in url_lower:
            return StakeGame.PLINKO
        elif "/limbo" in url_lower:
            return StakeGame.LIMBO
        elif "/keno" in url_lower:
            return StakeGame.KENO
        elif "/slots" in url_lower or "/casino/games/" in url_lower:
            return StakeGame.SLOTS
        else:
            return StakeGame.UNKNOWN

    def _extract_replay_id(self, url: str) -> str:
        """Extrae el ID del replay de la URL."""
        # Formato típico: ...?game=xxx&modal=replay&id=123
        # O: ...replay/123
        import re

        # Buscar patrón id=xxx
        match = re.search(r'[?&]id=([^&]+)', url)
        if match:
            return match.group(1)

        # Buscar patrón /replay/xxx
        match = re.search(r'/replay/([^/?]+)', url)
        if match:
            return match.group(1)

        # Fallback: usar timestamp
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    async def _init_browser(self) -> None:
        """Inicializa el navegador si no está iniciado."""
        if self._browser is None:
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(
                headless=self.options.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ]
            )

            self._context = await self._browser.new_context(
                viewport={
                    "width": self.options.width,
                    "height": self.options.height,
                },
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )

    async def close(self) -> None:
        """Cierra el navegador."""
        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

    async def capture(
        self,
        replay_url: str,
        duration: Optional[float] = None,
        options: Optional[CaptureOptions] = None,
        custom_filename: Optional[str] = None,
    ) -> Optional[ReplayInfo]:
        """
        Captura un replay de Stake como video.

        Args:
            replay_url: URL del replay
            duration: Duración de la captura (usa max_duration si no se especifica)
            options: Opciones de captura
            custom_filename: Nombre personalizado (sin extensión)

        Returns:
            ReplayInfo o None si falla
        """
        opts = options or self.options
        capture_duration = min(duration or opts.max_duration, opts.max_duration)

        self._stats["total_captures"] += 1

        try:
            await self._init_browser()

            # Crear página
            page = await self._context.new_page()

            # Navegar a la URL
            await page.goto(replay_url, wait_until="networkidle")

            # Esperar a que el juego cargue
            game = self._detect_game(replay_url)
            await self._wait_for_game_ready(page, game, opts.wait_for_load)

            # Generar nombre de archivo
            replay_id = self._extract_replay_id(replay_url)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if custom_filename:
                filename = f"{custom_filename}.mp4"
            else:
                filename = f"stake_{game.value}_{replay_id}_{timestamp}.mp4"

            output_path = self.output_dir / filename

            # Capturar video
            success = await self._record_screen(
                page,
                output_path,
                capture_duration,
                opts,
            )

            await page.close()

            if success and output_path.exists():
                self._stats["successful"] += 1
                self._stats["total_duration"] += capture_duration

                return ReplayInfo(
                    replay_id=replay_id,
                    game=game,
                    url=replay_url,
                    file_path=output_path,
                    duration=capture_duration,
                    resolution=(opts.width, opts.height),
                    metadata={"fps": opts.fps},
                )

            self._stats["failed"] += 1
            return None

        except Exception as e:
            self._stats["failed"] += 1
            return None

    async def _wait_for_game_ready(
        self,
        page: Page,
        game: StakeGame,
        timeout: float,
    ) -> None:
        """Espera a que el juego esté listo para reproducir."""
        try:
            selector = self.GAME_READY_SELECTORS.get(game)
            if selector:
                await page.wait_for_selector(selector, timeout=timeout * 1000)
            else:
                # Fallback: esperar tiempo fijo
                await asyncio.sleep(timeout)
        except Exception:
            # Si falla, esperar tiempo fijo
            await asyncio.sleep(timeout)

    async def _record_screen(
        self,
        page: Page,
        output_path: Path,
        duration: float,
        opts: CaptureOptions,
    ) -> bool:
        """
        Graba la pantalla usando capturas de Playwright + FFmpeg.

        Esta es una implementación simplificada. Para mejor calidad,
        considera usar herramientas como OBS + obs-websocket.
        """
        try:
            # Método 1: Usar screenshots en serie (más simple, menor calidad)
            # Para mejor calidad, ver _record_with_ffmpeg_pipe

            temp_dir = Path(tempfile.mkdtemp())
            frame_count = int(duration * opts.fps)
            frame_interval = 1.0 / opts.fps

            # Capturar frames
            for i in range(frame_count):
                frame_path = temp_dir / f"frame_{i:06d}.png"
                await page.screenshot(path=str(frame_path))
                await asyncio.sleep(frame_interval)

            # Combinar frames con FFmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Sobrescribir
                "-framerate", str(opts.fps),
                "-i", str(temp_dir / "frame_%06d.png"),
                "-c:v", opts.video_codec,
                "-crf", str(opts.crf),
                "-preset", opts.preset,
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await process.wait()

            # Limpiar frames temporales
            for frame in temp_dir.glob("*.png"):
                frame.unlink()
            temp_dir.rmdir()

            return output_path.exists()

        except Exception as e:
            return False

    async def _record_with_ffmpeg_pipe(
        self,
        page: Page,
        output_path: Path,
        duration: float,
        opts: CaptureOptions,
    ) -> bool:
        """
        Método alternativo: enviar frames directamente a FFmpeg via pipe.
        Más eficiente pero más complejo.
        """
        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{opts.width}x{opts.height}",
                "-r", str(opts.fps),
                "-i", "pipe:0",
                "-c:v", opts.video_codec,
                "-crf", str(opts.crf),
                "-preset", opts.preset,
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            frame_count = int(duration * opts.fps)
            frame_interval = 1.0 / opts.fps

            for _ in range(frame_count):
                screenshot = await page.screenshot(type="png")
                # Convertir PNG a raw RGB (requiere Pillow)
                # process.stdin.write(raw_data)
                await asyncio.sleep(frame_interval)

            process.stdin.close()
            await process.wait()

            return output_path.exists()

        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de capturas."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful"] / self._stats["total_captures"]
                if self._stats["total_captures"] > 0 else 0
            ),
        }


# =====================================================
# CAPTURA CON OBS (ALTERNATIVA RECOMENDADA)
# =====================================================

class OBSReplayCapture:
    """
    Captura de replays usando OBS Studio.

    Más eficiente y mejor calidad que capturas de Playwright.
    Requiere OBS + obs-websocket plugin.

    NOTA: Esta es una implementación placeholder.
    Para implementación completa, usar obsws-python.
    """

    def __init__(
        self,
        obs_host: str = "localhost",
        obs_port: int = 4455,
        obs_password: Optional[str] = None,
    ):
        self.host = obs_host
        self.port = obs_port
        self.password = obs_password
        self._connected = False

    async def connect(self) -> bool:
        """Conecta a OBS via WebSocket."""
        # TODO: Implementar con obsws-python
        # from obsws_python import ReqClient
        # self.client = ReqClient(host=self.host, port=self.port, password=self.password)
        return False

    async def start_recording(self) -> bool:
        """Inicia grabación en OBS."""
        # self.client.start_record()
        return False

    async def stop_recording(self) -> Optional[str]:
        """Detiene grabación y retorna path del archivo."""
        # self.client.stop_record()
        return None

    async def save_replay_buffer(self) -> Optional[str]:
        """Guarda el Replay Buffer de OBS."""
        # self.client.save_replay_buffer()
        return None


# =====================================================
# EJEMPLO DE USO
# =====================================================

async def main():
    """Ejemplo de uso del ReplayRenderer."""
    from core.logger import logger

    if not HAS_PLAYWRIGHT:
        logger.error("Playwright no instalado. Ejecuta: pip install playwright")
        return

    output_dir = Path("./replays")
    renderer = ReplayRenderer(
        output_dir=output_dir,
        default_options=CaptureOptions(
            width=1280,
            height=720,
            fps=30,
            max_duration=30.0,
            headless=False,  # Ver navegador para debug
        ),
    )

    # URL de ejemplo (reemplazar con URL real de replay)
    test_url = "https://stake.com/casino/games/crash?modal=replay&id=example"

    logger.info(f"Capturando replay: {test_url}")
    logger.warning("NOTA: Esto requiere una URL real de replay de Stake")

    # Descomentar para probar con URL real
    # replay = await renderer.capture(test_url, duration=15.0)
    # if replay:
    #     logger.info(f"Capturado: {replay.file_path}")

    await renderer.close()
    logger.info(f"Stats: {renderer.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
