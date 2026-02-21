"""
YouTube Shorts Uploader - Sube videos a YouTube Shorts.

Usa la API oficial de YouTube Data API v3.
Requiere:
1. Crear proyecto en Google Cloud Console
2. Habilitar YouTube Data API v3
3. Crear credenciales OAuth 2.0
4. Descargar client_secrets.json

Documentación: https://developers.google.com/youtube/v3
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json
import os

# Intentar importar google-auth
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False


@dataclass
class YouTubeUploadResult:
    """Resultado de un upload a YouTube."""
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def shorts_url(self) -> Optional[str]:
        if self.video_id:
            return f"https://youtube.com/shorts/{self.video_id}"
        return None


@dataclass
class YouTubeConfig:
    """Configuración para YouTube uploader."""
    client_secrets_file: Path = Path("./client_secrets.json")
    token_file: Path = Path("./youtube_token.json")
    scopes: List[str] = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = ["https://www.googleapis.com/auth/youtube.upload"]


class YouTubeShortsUploader:
    """
    Uploader de videos a YouTube Shorts usando la API oficial.

    Requisitos:
    1. pip install google-auth google-auth-oauthlib google-api-python-client
    2. Crear proyecto en Google Cloud Console
    3. Habilitar YouTube Data API v3
    4. Crear credenciales OAuth 2.0 (tipo: Desktop app)
    5. Descargar client_secrets.json

    Uso:
        uploader = YouTubeShortsUploader()
        await uploader.authenticate()
        result = await uploader.upload(
            video_path="video.mp4",
            title="Mi Short viral",
            description="Descripción del video",
        )
    """

    def __init__(self, config: Optional[YouTubeConfig] = None):
        self.config = config or YouTubeConfig()
        self._credentials: Optional[Credentials] = None
        self._youtube = None

    def authenticate(self) -> bool:
        """
        Autentica con YouTube API.

        La primera vez abrirá el navegador para autorizar.
        Las siguientes veces usará el token guardado.

        Returns:
            True si la autenticación fue exitosa
        """
        if not HAS_GOOGLE_API:
            print("ERROR: google-api-python-client no instalado")
            print("Ejecuta: pip install google-auth google-auth-oauthlib google-api-python-client")
            return False

        if not self.config.client_secrets_file.exists():
            print(f"ERROR: No se encontró {self.config.client_secrets_file}")
            print("")
            print("Para configurar YouTube API:")
            print("1. Ve a https://console.cloud.google.com/")
            print("2. Crea un proyecto nuevo")
            print("3. Habilita 'YouTube Data API v3'")
            print("4. Ve a Credenciales > Crear credenciales > ID de cliente OAuth")
            print("5. Selecciona 'Aplicación de escritorio'")
            print("6. Descarga el JSON y guárdalo como 'client_secrets.json'")
            return False

        try:
            creds = None

            # Cargar token existente
            if self.config.token_file.exists():
                creds = Credentials.from_authorized_user_file(
                    str(self.config.token_file),
                    self.config.scopes
                )

            # Si no hay credenciales válidas, autenticar
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.config.client_secrets_file),
                        self.config.scopes
                    )
                    creds = flow.run_local_server(port=0)

                # Guardar token
                with open(self.config.token_file, 'w') as token:
                    token.write(creds.to_json())

            self._credentials = creds
            self._youtube = build('youtube', 'v3', credentials=creds)

            return True

        except Exception as e:
            print(f"Error de autenticación: {e}")
            return False

    async def upload(
        self,
        video_path: str | Path,
        title: str,
        description: str = "",
        tags: List[str] = None,
        category_id: str = "22",  # 22 = People & Blogs
        privacy: str = "public",  # public, private, unlisted
        is_short: bool = True,
    ) -> YouTubeUploadResult:
        """
        Sube un video a YouTube.

        Args:
            video_path: Ruta al archivo de video
            title: Título del video (max 100 caracteres)
            description: Descripción del video
            tags: Lista de tags
            category_id: ID de categoría de YouTube
            privacy: Privacidad (public, private, unlisted)
            is_short: Si es un Short (agrega #Shorts al título)

        Returns:
            YouTubeUploadResult con el resultado
        """
        video_path = Path(video_path)

        if not video_path.exists():
            return YouTubeUploadResult(
                success=False,
                error=f"Video no encontrado: {video_path}"
            )

        if not self._youtube:
            return YouTubeUploadResult(
                success=False,
                error="No autenticado. Usa authenticate() primero."
            )

        # Para Shorts, agregar #Shorts al título
        if is_short and "#Shorts" not in title:
            title = f"{title} #Shorts"

        # Truncar título si es muy largo
        if len(title) > 100:
            title = title[:97] + "..."

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags or [],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            }
        }

        try:
            # Ejecutar upload en thread separado (es bloqueante)
            result = await asyncio.to_thread(
                self._upload_video,
                str(video_path),
                body
            )

            if result and "id" in result:
                return YouTubeUploadResult(
                    success=True,
                    video_id=result["id"],
                    video_url=f"https://youtube.com/watch?v={result['id']}",
                )
            else:
                return YouTubeUploadResult(
                    success=False,
                    error="Upload failed - no video ID returned"
                )

        except Exception as e:
            return YouTubeUploadResult(
                success=False,
                error=str(e)
            )

    def _upload_video(self, video_path: str, body: dict) -> dict:
        """Upload sincrónico (para ejecutar en thread)."""
        media = MediaFileUpload(
            video_path,
            mimetype="video/mp4",
            resumable=True,
            chunksize=1024 * 1024  # 1MB chunks
        )

        request = self._youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")

        return response

    async def get_channel_info(self) -> Optional[Dict]:
        """Obtiene información del canal autenticado."""
        if not self._youtube:
            return None

        try:
            response = self._youtube.channels().list(
                part="snippet,statistics",
                mine=True
            ).execute()

            if response.get("items"):
                channel = response["items"][0]
                return {
                    "id": channel["id"],
                    "title": channel["snippet"]["title"],
                    "subscribers": channel["statistics"].get("subscriberCount", 0),
                }
        except Exception:
            pass

        return None


# ============================================
# EJEMPLO DE USO
# ============================================

async def main():
    """Ejemplo de uso del YouTubeShortsUploader."""
    print("=== YouTube Shorts Uploader ===")
    print("")

    if not HAS_GOOGLE_API:
        print("ERROR: Bibliotecas de Google no instaladas")
        print("Ejecuta: pip install google-auth google-auth-oauthlib google-api-python-client")
        return

    uploader = YouTubeShortsUploader()

    # Autenticar
    if uploader.authenticate():
        print("[OK] Autenticado con YouTube")

        # Obtener info del canal
        channel = await uploader.get_channel_info()
        if channel:
            print(f"Canal: {channel['title']}")
            print(f"Suscriptores: {channel['subscribers']}")

        # Ejemplo de upload (descomentar para usar)
        # result = await uploader.upload(
        #     video_path="./clips/video.mp4",
        #     title="Momento viral increíble",
        #     description="Un momento épico capturado en stream",
        #     tags=["gaming", "viral", "win", "shorts"],
        # )
        # print(f"Upload: {result.success}")
        # if result.success:
        #     print(f"URL: {result.shorts_url}")
    else:
        print("[ERROR] No se pudo autenticar")


if __name__ == "__main__":
    asyncio.run(main())
