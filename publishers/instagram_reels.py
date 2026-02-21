"""
Instagram Reels Uploader - Sube videos a Instagram Reels.

Instagram tiene restricciones estrictas:
1. Instagram Graph API (oficial) - Solo para cuentas business/creator
2. Instagrapi (no oficial) - Funciona pero puede resultar en bans

Este módulo usa instagrapi para automatización.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json

# Intentar importar instagrapi
try:
    from instagrapi import Client
    from instagrapi.types import Media
    HAS_INSTAGRAPI = True
except ImportError:
    HAS_INSTAGRAPI = False
    Client = None


@dataclass
class InstagramUploadResult:
    """Resultado de un upload a Instagram."""
    success: bool
    media_id: Optional[str] = None
    media_url: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class InstagramConfig:
    """Configuración para Instagram uploader."""
    session_file: Path = Path("./instagram_session.json")
    username: Optional[str] = None
    password: Optional[str] = None


class InstagramReelsUploader:
    """
    Uploader de Reels a Instagram usando instagrapi.

    ADVERTENCIA:
    - Instagram puede banear cuentas que usan automatización
    - Usar con moderación y delays apropiados
    - Considerar usar cuentas secundarias para pruebas

    Requisitos:
    - pip install instagrapi

    Uso:
        uploader = InstagramReelsUploader(
            config=InstagramConfig(
                username="tu_usuario",
                password="tu_password",
            )
        )
        await uploader.login()
        result = await uploader.upload_reel(
            video_path="video.mp4",
            caption="Mi reel viral #fyp",
        )
    """

    def __init__(self, config: Optional[InstagramConfig] = None):
        if not HAS_INSTAGRAPI:
            raise ImportError(
                "instagrapi no está instalado. "
                "Ejecuta: pip install instagrapi"
            )

        self.config = config or InstagramConfig()
        self._client: Optional[Client] = None
        self._logged_in = False

    async def login(self) -> bool:
        """
        Inicia sesión en Instagram.

        Intenta cargar sesión guardada primero.

        Returns:
            True si el login fue exitoso
        """
        if not self.config.username or not self.config.password:
            print("ERROR: Necesitas configurar username y password")
            return False

        try:
            self._client = Client()

            # Intentar cargar sesión existente
            if self.config.session_file.exists():
                try:
                    self._client.load_settings(str(self.config.session_file))
                    self._client.login(self.config.username, self.config.password)
                    self._logged_in = True
                    return True
                except Exception:
                    # Sesión inválida, hacer login nuevo
                    pass

            # Login nuevo
            result = await asyncio.to_thread(
                self._client.login,
                self.config.username,
                self.config.password
            )

            if result:
                # Guardar sesión
                self._client.dump_settings(str(self.config.session_file))
                self._logged_in = True
                return True

        except Exception as e:
            print(f"Error de login: {e}")

        return False

    async def upload_reel(
        self,
        video_path: str | Path,
        caption: str = "",
        hashtags: List[str] = None,
        thumbnail_path: Optional[str | Path] = None,
    ) -> InstagramUploadResult:
        """
        Sube un Reel a Instagram.

        Args:
            video_path: Ruta al archivo de video (MP4, max 90 seg)
            caption: Caption del reel
            hashtags: Lista de hashtags (sin #)
            thumbnail_path: Ruta a imagen de thumbnail (opcional)

        Returns:
            InstagramUploadResult con el resultado
        """
        video_path = Path(video_path)

        if not video_path.exists():
            return InstagramUploadResult(
                success=False,
                error=f"Video no encontrado: {video_path}"
            )

        if not self._logged_in or not self._client:
            return InstagramUploadResult(
                success=False,
                error="No has iniciado sesión. Usa login() primero."
            )

        # Construir caption con hashtags
        full_caption = caption
        if hashtags:
            tags = " ".join(f"#{tag}" for tag in hashtags)
            full_caption = f"{caption}\n\n{tags}"

        try:
            # Upload en thread separado (es bloqueante)
            if thumbnail_path and Path(thumbnail_path).exists():
                media = await asyncio.to_thread(
                    self._client.clip_upload,
                    str(video_path),
                    full_caption,
                    str(thumbnail_path),
                )
            else:
                media = await asyncio.to_thread(
                    self._client.clip_upload,
                    str(video_path),
                    full_caption,
                )

            if media:
                return InstagramUploadResult(
                    success=True,
                    media_id=media.pk,
                    media_url=f"https://www.instagram.com/reel/{media.code}/",
                )
            else:
                return InstagramUploadResult(
                    success=False,
                    error="Upload failed - no media returned"
                )

        except Exception as e:
            return InstagramUploadResult(
                success=False,
                error=str(e)
            )

    async def get_account_info(self) -> Optional[Dict]:
        """Obtiene información de la cuenta."""
        if not self._logged_in or not self._client:
            return None

        try:
            user = await asyncio.to_thread(self._client.account_info)
            return {
                "username": user.username,
                "full_name": user.full_name,
                "followers": user.follower_count,
                "following": user.following_count,
                "posts": user.media_count,
            }
        except Exception:
            pass

        return None

    def logout(self) -> None:
        """Cierra la sesión."""
        if self._client:
            try:
                self._client.logout()
            except Exception:
                pass
            self._client = None
        self._logged_in = False


# ============================================
# EJEMPLO DE USO
# ============================================

async def main():
    """Ejemplo de uso del InstagramReelsUploader."""
    print("=== Instagram Reels Uploader ===")
    print("")

    if not HAS_INSTAGRAPI:
        print("ERROR: instagrapi no instalado")
        print("Ejecuta: pip install instagrapi")
        return

    # IMPORTANTE: Configura tus credenciales
    config = InstagramConfig(
        username="TU_USUARIO",  # Cambiar
        password="TU_PASSWORD",  # Cambiar
    )

    uploader = InstagramReelsUploader(config=config)

    try:
        if await uploader.login():
            print("[OK] Logueado en Instagram")

            # Info de cuenta
            info = await uploader.get_account_info()
            if info:
                print(f"Usuario: {info['username']}")
                print(f"Seguidores: {info['followers']}")

            # Ejemplo de upload (descomentar para usar)
            # result = await uploader.upload_reel(
            #     video_path="./clips/video.mp4",
            #     caption="Momento viral increíble",
            #     hashtags=["viral", "gaming", "reels", "fyp"],
            # )
            # print(f"Upload: {result.success}")
            # if result.success:
            #     print(f"URL: {result.media_url}")
        else:
            print("[ERROR] No se pudo iniciar sesión")

    finally:
        uploader.logout()


if __name__ == "__main__":
    asyncio.run(main())
