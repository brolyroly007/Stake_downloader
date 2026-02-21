"""
TikTok Uploader - Sube videos a TikTok automáticamente.

NOTA: TikTok no tiene API pública oficial para subir videos.
Opciones disponibles:
1. TikTok for Business API (requiere cuenta de negocio aprobada)
2. Selenium/Playwright (automatización del navegador)
3. Servicios de terceros (tikhub.io, etc.)

Este módulo usa Playwright para automatización del navegador.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json

# Intentar importar Playwright
try:
    from playwright.async_api import async_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    Browser = None
    Page = None


@dataclass
class TikTokUploadResult:
    """Resultado de un upload a TikTok."""
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TikTokConfig:
    """Configuración para TikTok uploader."""
    session_file: Path = Path("./tiktok_session.json")
    headless: bool = False  # False para ver el navegador (útil para login)
    timeout: int = 60000  # ms


class TikTokUploader:
    """
    Uploader de videos a TikTok usando automatización de navegador.

    IMPORTANTE:
    - Requiere login manual la primera vez
    - La sesión se guarda para futuros uploads
    - TikTok puede detectar bots, usar con moderación

    Uso:
        uploader = TikTokUploader()
        await uploader.login()  # Primera vez: login manual
        result = await uploader.upload(
            video_path="video.mp4",
            caption="Mi video viral #fyp",
        )
    """

    TIKTOK_URL = "https://www.tiktok.com"
    UPLOAD_URL = "https://www.tiktok.com/creator-center/upload"

    def __init__(self, config: Optional[TikTokConfig] = None):
        if not HAS_PLAYWRIGHT:
            raise ImportError(
                "Playwright no está instalado. "
                "Ejecuta: pip install playwright && playwright install chromium"
            )

        self.config = config or TikTokConfig()
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._logged_in = False

    async def _init_browser(self) -> None:
        """Inicializa el navegador."""
        if self._browser is None:
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(
                headless=self.config.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ]
            )

            # Cargar contexto con sesión guardada si existe
            context_options = {
                "viewport": {"width": 1280, "height": 720},
                "user_agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }

            if self.config.session_file.exists():
                context_options["storage_state"] = str(self.config.session_file)

            context = await self._browser.new_context(**context_options)
            self._page = await context.new_page()

    async def close(self) -> None:
        """Cierra el navegador."""
        if self._page:
            # Guardar sesión
            await self._page.context.storage_state(path=str(self.config.session_file))
            await self._page.close()
            self._page = None

        if self._browser:
            await self._browser.close()
            self._browser = None

    async def login(self, wait_for_manual: bool = True) -> bool:
        """
        Inicia sesión en TikTok.

        Si wait_for_manual=True, espera a que el usuario complete el login manualmente.

        Returns:
            True si el login fue exitoso
        """
        await self._init_browser()

        # Ir a TikTok
        await self._page.goto(self.TIKTOK_URL, wait_until="networkidle")

        # Verificar si ya está logueado
        if await self._is_logged_in():
            self._logged_in = True
            return True

        if wait_for_manual:
            print("")
            print("=" * 50)
            print("  LOGIN MANUAL REQUERIDO")
            print("=" * 50)
            print("")
            print("1. Se abrió una ventana de navegador")
            print("2. Inicia sesión en TikTok manualmente")
            print("3. Cuando termines, presiona Enter aquí")
            print("")
            input("Presiona Enter cuando hayas iniciado sesión...")

            # Verificar login
            if await self._is_logged_in():
                self._logged_in = True
                await self._page.context.storage_state(path=str(self.config.session_file))
                print("[OK] Sesión guardada")
                return True

        return False

    async def _is_logged_in(self) -> bool:
        """Verifica si el usuario está logueado."""
        try:
            # Buscar elementos que indican login
            await self._page.goto(self.UPLOAD_URL, wait_until="networkidle", timeout=10000)

            # Si no redirige a login, está logueado
            current_url = self._page.url
            return "login" not in current_url.lower()

        except Exception:
            return False

    async def upload(
        self,
        video_path: str | Path,
        caption: str = "",
        tags: List[str] = None,
        schedule: Optional[datetime] = None,
    ) -> TikTokUploadResult:
        """
        Sube un video a TikTok.

        Args:
            video_path: Ruta al archivo de video
            caption: Descripción del video
            tags: Lista de hashtags (sin #)
            schedule: Fecha/hora para publicar (None = inmediato)

        Returns:
            TikTokUploadResult con el resultado
        """
        video_path = Path(video_path)

        if not video_path.exists():
            return TikTokUploadResult(
                success=False,
                error=f"Video no encontrado: {video_path}"
            )

        if not self._logged_in:
            return TikTokUploadResult(
                success=False,
                error="No has iniciado sesión. Usa login() primero."
            )

        try:
            await self._init_browser()

            # Ir a página de upload
            await self._page.goto(self.UPLOAD_URL, wait_until="networkidle")

            # Esperar iframe de upload
            await self._page.wait_for_selector('iframe', timeout=self.config.timeout)

            # Subir archivo
            # TikTok usa un input file dentro de un iframe
            file_input = await self._page.query_selector('input[type="file"]')

            if file_input:
                await file_input.set_input_files(str(video_path))
            else:
                # Intentar con el selector del iframe
                frame = self._page.frame_locator('iframe').first
                file_input = frame.locator('input[type="file"]')
                await file_input.set_input_files(str(video_path))

            # Esperar que suba
            await asyncio.sleep(5)

            # Agregar caption
            full_caption = caption
            if tags:
                hashtags = " ".join(f"#{tag}" for tag in tags)
                full_caption = f"{caption} {hashtags}"

            # Buscar campo de caption
            caption_input = await self._page.query_selector('[data-e2e="caption-input"]')
            if caption_input:
                await caption_input.fill(full_caption)

            # Esperar procesamiento
            await asyncio.sleep(10)

            # Click en publicar
            post_button = await self._page.query_selector('[data-e2e="post-button"]')
            if post_button:
                await post_button.click()

            # Esperar confirmación
            await asyncio.sleep(5)

            return TikTokUploadResult(
                success=True,
                video_url=self._page.url,
            )

        except Exception as e:
            return TikTokUploadResult(
                success=False,
                error=str(e)
            )


# ============================================
# EJEMPLO DE USO
# ============================================

async def main():
    """Ejemplo de uso del TikTokUploader."""
    print("=== TikTok Uploader ===")
    print("")

    if not HAS_PLAYWRIGHT:
        print("ERROR: Playwright no instalado")
        print("Ejecuta: pip install playwright && playwright install chromium")
        return

    uploader = TikTokUploader(
        config=TikTokConfig(
            headless=False,  # Ver navegador para login
        )
    )

    try:
        # Login (primera vez requiere manual)
        logged_in = await uploader.login()

        if logged_in:
            print("[OK] Logueado en TikTok")

            # Ejemplo de upload (descomentar para usar)
            # result = await uploader.upload(
            #     video_path="./clips/video.mp4",
            #     caption="Momento viral increíble",
            #     tags=["fyp", "viral", "gaming", "win"],
            # )
            # print(f"Upload: {result.success}")
        else:
            print("[ERROR] No se pudo iniciar sesión")

    finally:
        await uploader.close()


if __name__ == "__main__":
    asyncio.run(main())
