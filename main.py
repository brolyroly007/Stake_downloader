"""
Viral Clip Monitor - Aplicación Web
Dashboard minimalista para monitorear streams y detectar momentos virales.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Optional
from pathlib import Path
import sys
import os
import multiprocessing

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

# Importar monitores
# Asegurar que el directorio actual está en path
if getattr(sys, 'frozen', False):
    base_dir = Path(sys._MEIPASS)
else:
    base_dir = Path(__file__).parent

sys.path.insert(0, str(base_dir))

from monitors.kick_monitor import KickMonitor
from monitors.base_monitor import EventType, MonitorEvent
from pipeline import ViralPipeline
import config

# Configurar logger para enviar a WebSocket
from core.logger import logger

def websocket_sink(message):
    """Sink para enviar logs a WebSocket."""
    # message es un objeto Message de loguru, convertir a string
    text = message.record["message"]
    # Usar el loop actual para enviar el mensaje
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.create_task(broadcast_log(f"{message.record['time'].strftime('%H:%M:%S')} | {text}"))
    except RuntimeError:
        pass

# Añadir sink
logger.add(websocket_sink, format="{message}", level="INFO")

# ============================================
# CONFIGURACIÓN
# ============================================

app = FastAPI(title="Viral Clip Monitor", version="1.0.0")

# Pipeline de procesamiento
pipeline = ViralPipeline(
    output_dir=Path("./clips"),
    auto_download=True,
)

# Estado global
state = {
    "kick_monitor": None,
    "pipeline": pipeline,
    "is_running": False,
    "channels": config.MONITORED_CHANNELS,
    "events": [],  # Últimos eventos
    "stats": {
        "messages": 0,
        "spikes": 0,
        "clips_ready": 0,
        "start_time": None,
    }
}

# WebSocket connections para updates en tiempo real
connected_clients: Set[WebSocket] = set()


# ============================================
# WEBSOCKET MANAGER
# ============================================

async def broadcast(data: dict):
    """Envía datos a todos los clientes conectados."""
    if not connected_clients:
        return
    message = json.dumps(data, default=str)
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_text(message)
        except:
            disconnected.add(client)
    connected_clients.difference_update(disconnected)


async def broadcast_log(message: str):
    """Envía logs a los clientes."""
    await broadcast({"type": "log", "data": message})

# ============================================
# EVENT HANDLERS
# ============================================

async def on_chat_message(event: MonitorEvent):
    """Handler para mensajes de chat."""
    state["stats"]["messages"] += 1

    # Solo broadcast cada 10 mensajes para no saturar
    if state["stats"]["messages"] % 10 == 0:
        await broadcast({
            "type": "stats",
            "data": state["stats"]
        })


async def on_chat_spike(event: MonitorEvent):
    """Handler para spikes de chat."""
    state["stats"]["spikes"] += 1

    event_data = {
        "type": "spike",
        "channel": event.channel,
        "velocity": event.data.get("velocity", 0),
        "timestamp": datetime.now().isoformat(),
    }

    state["events"].insert(0, event_data)
    state["events"] = state["events"][:50]  # Mantener últimos 50

    await broadcast({
        "type": "event",
        "data": event_data
    })

    # Enviar al pipeline para descarga automática
    await pipeline.handle_event(event)


async def on_clip_ready(moment):
    """Callback cuando un clip está listo."""
    state["stats"]["clips_ready"] += 1
    await broadcast({
        "type": "clip_ready",
        "data": {
            "id": moment.id,
            "channel": moment.channel,
            "type": moment.event_type,
            "path": str(moment.clip_path) if moment.clip_path else None,
            "timestamp": moment.timestamp.isoformat(),
        }
    })

async def update_stats():
    """Actualiza estadísticas periódicamente."""
    while True:
        if state["kick_monitor"]:
            # Obtener velocidades por canal
            velocities = state["kick_monitor"].get_current_velocities()
            
            # Actualizar estado
            state["stats"]["messages"] = state["kick_monitor"].total_messages
            
            # Enviar update a clientes
            await broadcast({
                "type": "stats",
                "data": {
                    **state["stats"],
                    "velocities": velocities  # Enviar dict {canal: velocidad}
                }
            })
            
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_stats())

# ============================================
# API ROUTES
# ============================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Página principal."""
    # Manejo de rutas para PyInstaller
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent

    return FileResponse(base_path / "web" / "index.html")


@app.get("/api/status")
async def get_status():
    """Estado actual del sistema."""
    channels_info = []
    if state["kick_monitor"]:
        for slug, channel in state["kick_monitor"].channels.items():
            channels_info.append({
                "slug": slug,
                "chatroom_id": channel.chatroom_id,
                "is_live": channel.is_live,
                "viewer_count": channel.viewer_count,
                "velocity": state["kick_monitor"].get_chat_velocity(slug),
            })

    return {
        "is_running": state["is_running"],
        "channels": channels_info,
        "stats": state["stats"],
        "events": state["events"][:20],
    }


@app.post("/api/start")
async def start_monitoring():
    """Inicia el monitoreo."""
    if state["is_running"]:
        return {"status": "already_running"}

    # Crear monitor de Kick
    # Reiniciar si ya existe para aplicar cambios de canales
    if state["kick_monitor"]:
        await state["kick_monitor"].stop()

    state["kick_monitor"] = KickMonitor(
        channels=state["channels"],
        chat_velocity_threshold=5,
        detect_chat_spikes=True,
    )

    # Registrar handlers
    state["kick_monitor"].on_event(EventType.CHAT_MESSAGE, on_chat_message)
    state["kick_monitor"].on_event(EventType.CHAT_SPIKE, on_chat_spike)

    # Registrar callback del pipeline
    pipeline.on_ready(on_clip_ready)

    # Conectar
    kick_ok = await state["kick_monitor"].connect()

    if kick_ok:
        await state["kick_monitor"].start()

    state["is_running"] = True
    state["stats"]["start_time"] = datetime.now().isoformat()

    await broadcast({"type": "status", "data": {"is_running": True}})

    return {
        "status": "started",
        "kick": kick_ok,
    }


@app.post("/api/stop")
async def stop_monitoring():
    """Detiene el monitoreo."""
    if not state["is_running"]:
        return {"status": "not_running"}

    if state["kick_monitor"]:
        await state["kick_monitor"].stop()

    state["is_running"] = False

    await broadcast({"type": "status", "data": {"is_running": False}})

    return {"status": "stopped"}


@app.post("/api/channels")
async def set_channels(channels: List[str]):
    """Actualiza los canales a monitorear."""
    state["channels"] = channels
    # Actualizar config en memoria
    config.MONITORED_CHANNELS = channels
    
    # Si está corriendo, reiniciar el monitor para aplicar cambios
    if state["is_running"] and state["kick_monitor"]:
        await state["kick_monitor"].stop()
        await start_monitoring()
        
    return {"status": "updated", "channels": channels}

@app.get("/api/config")
async def get_config():
    """Obtiene la configuración actual."""
    return {
        "channels": state["channels"],
        "thresholds": {
            "chat_velocity": config.CHAT_VELOCITY_THRESHOLD,
            "virality": config.VIRALITY_THRESHOLD,
        },
        "tiktok_enabled": config.ENABLE_TIKTOK_UPLOAD,
    }


@app.get("/api/clips")
async def get_clips():
    """Lista de clips capturados."""
    moments = pipeline.get_recent_moments(limit=50)
    stats = pipeline.get_stats()
    return {
        "clips": moments,
        "stats": stats,
    }


@app.get("/api/clips/{clip_id}")
async def get_clip(clip_id: str):
    """Obtiene un clip específico."""
    for moment in pipeline.moments:
        if moment.id == clip_id:
            return {
                "id": moment.id,
                "type": moment.event_type,
                "channel": moment.channel,
                "timestamp": moment.timestamp.isoformat(),
                "status": moment.status,
                "clip_url": moment.clip_url,
                "clip_path": str(moment.clip_path) if moment.clip_path else None,
                "processed_path": str(moment.processed_path) if moment.processed_path else None,
                "published_urls": moment.published_urls,
                "data": moment.data,
            }
    return {"error": "Clip not found"}, 404


@app.post("/api/capture/{channel}")
async def capture_clip(channel: str, duration: int = 30):
    """Captura un clip manualmente de un canal."""
    from monitors.base_monitor import MonitorEvent, EventType

    # Crear evento manual
    event = MonitorEvent(
        event_type=EventType.CHAT_SPIKE,
        source="kick",
        channel=channel,
        timestamp=datetime.now(),
        data={"velocity": 0, "manual": True},
    )

    # Enviar al pipeline
    moment = await pipeline.handle_event(event)

    if moment:
        return {
            "status": "capturing",
            "clip_id": moment.id,
            "channel": channel,
        }
    return {"status": "error", "error": "Failed to create capture"}


@app.post("/api/publish/{clip_id}/{platform}")
async def publish_clip(clip_id: str, platform: str):
    """Publica un clip a una plataforma."""
    # Buscar el momento
    moment = None
    for m in pipeline.moments:
        if m.id == clip_id:
            moment = m
            break

    if not moment:
        return {"error": "Clip not found"}, 404

    if not moment.clip_path or not moment.clip_path.exists():
        return {"error": "Clip file not found"}, 404

    # Publicar según la plataforma
    result = None
    error_msg = "Platform not configured"

    try:
        if platform == "tiktok":
            from publishers import TikTokUploader, TikTokConfig
            uploader = TikTokUploader(TikTokConfig(headless=True))
            # Nota: TikTok requiere login manual previo para guardar sesión
            if await uploader.login(wait_for_manual=False):
                res = await uploader.upload(
                    video_path=moment.processed_path or moment.clip_path,
                    caption=f"Momento viral de {moment.channel} #fyp #gaming",
                    tags=["fyp", "viral", "gaming", "win", moment.channel],
                )
                if res.success:
                    moment.published_urls["tiktok"] = res.video_url
                    result = res
                else:
                    error_msg = res.error
            else:
                error_msg = "TikTok login failed (session not found)"
            await uploader.close()

        elif platform == "youtube":
            from publishers import YouTubeShortsUploader, YouTubeConfig
            uploader = YouTubeShortsUploader()
            if uploader.authenticate():
                res = await uploader.upload(
                    video_path=moment.processed_path or moment.clip_path,
                    title=f"Viral Moment: {moment.channel}",
                    description=f"Captured live from {moment.channel}\n\n#shorts #gaming #{moment.channel}",
                    tags=["gaming", "shorts", "viral", moment.channel],
                )
                if res.success:
                    moment.published_urls["youtube"] = res.shorts_url
                    result = res
                else:
                    error_msg = res.error
            else:
                error_msg = "YouTube authentication failed"

        elif platform == "instagram":
            from publishers import InstagramReelsUploader, InstagramConfig
            import config as app_config
            
            # Usar credenciales de config/env
            conf = InstagramConfig(
                username=getattr(app_config, 'INSTAGRAM_USERNAME', None),
                password=getattr(app_config, 'INSTAGRAM_PASSWORD', None)
            )
            
            uploader = InstagramReelsUploader(conf)
            if await uploader.login():
                res = await uploader.upload_reel(
                    video_path=moment.processed_path or moment.clip_path,
                    caption=f"Momento viral de {moment.channel} 🎮\n.\n.\n#gaming #viral #clips #{moment.channel}",
                    hashtags=["gaming", "viral", "clips", moment.channel]
                )
                if res.success:
                    moment.published_urls["instagram"] = res.media_url
                    result = res
                else:
                    error_msg = res.error
            else:
                error_msg = "Instagram login failed"
            
            uploader.logout()

    except Exception as e:
        error_msg = str(e)

    if result and result.success:
        return {"status": "ok", "url": getattr(result, 'video_url', None) or getattr(result, 'shorts_url', None) or getattr(result, 'media_url', None)}
    
    return {"status": "error", "error": error_msg}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para actualizaciones en tiempo real."""
    await websocket.accept()
    connected_clients.add(websocket)

    # Enviar estado inicial
    await websocket.send_text(json.dumps({
        "type": "init",
        "data": {
            "is_running": state["is_running"],
            "stats": state["stats"],
            "events": state["events"][:20],
        }
    }, default=str))

    try:
        while True:
            # Mantener conexión viva
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Necesario para PyInstaller
    multiprocessing.freeze_support()
    
    # Asegurar que estamos en el directorio correcto
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    args = parser.parse_args()

    print("")
    print("=" * 50)
    print("  VIRAL CLIP MONITOR")
    print("=" * 50)
    print("")
    print(f"  Abriendo en: http://localhost:{args.port}")
    print("")
    print("  Ctrl+C para detener")
    print("=" * 50)
    print("")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
