"""
Viral Clip Monitor - Aplicación Web
Dashboard minimalista para monitorear streams y detectar momentos virales.

Ejecutar: python app.py
Abrir: http://localhost:8000
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Optional
from pathlib import Path
import sys
import os
import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

# Importar monitores
import sys
sys.path.insert(0, str(Path(__file__).parent))

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
    "events": [],
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

    # Obtener velocidades de todos los canales monitoreados
    velocities = {}
    if state["kick_monitor"]:
        velocities = state["kick_monitor"].get_all_velocities()

    # Broadcast stats con velocidades cada 5 mensajes
    if state["stats"]["messages"] % 5 == 0:
        await broadcast({
            "type": "stats",
            "data": {
                **state["stats"],
                "velocities": velocities,
                "monitored_channels": list(state["channels"]),
            }
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


async def on_clip_analyzed(moment):
    """Callback cuando un clip es analizado por AI (antes de filtrar)."""
    # Actualizar resumen del stream con AI
    await update_stream_summary(moment.channel, moment)

    # Broadcast del análisis
    await broadcast({
        "type": "clip_analyzed",
        "data": {
            "id": moment.id,
            "channel": moment.channel,
            "ai_score": moment.ai_score,
            "ai_title": moment.ai_title,
            "ai_tags": moment.ai_tags,
            "is_viral": moment.ai_score >= 0.6,
            "timestamp": moment.timestamp.isoformat(),
        }
    })


async def on_clip_ready(moment):
    """Callback cuando un clip está listo (pasó filtro AI)."""
    state["stats"]["clips_ready"] += 1

    await broadcast({
        "type": "clip_ready",
        "data": {
            "id": moment.id,
            "channel": moment.channel,
            "type": moment.event_type,
            "path": str(moment.clip_path) if moment.clip_path else None,
            "timestamp": moment.timestamp.isoformat(),
            "ai_score": moment.ai_score,
            "ai_title": moment.ai_title,
            "ai_tags": moment.ai_tags,
        }
    })


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
    try:
        if state["kick_monitor"] and hasattr(state["kick_monitor"], 'channels'):
            for slug, channel in state["kick_monitor"].channels.items():
                channels_info.append({
                    "slug": slug,
                    "chatroom_id": getattr(channel, 'chatroom_id', 0),
                    "is_live": getattr(channel, 'is_live', False),
                    "viewer_count": getattr(channel, 'viewer_count', 0),
                    "velocity": state["kick_monitor"].get_chat_velocity(slug) if hasattr(state["kick_monitor"], 'get_chat_velocity') else 0,
                })
    except Exception as e:
        print(f"[Status] Error getting channels: {e}")

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
        chat_velocity_threshold=config.CHAT_VELOCITY_THRESHOLD,  # Usar config (default 15 msg/s)
        detect_chat_spikes=True,
    )

    # Registrar handlers
    state["kick_monitor"].on_event(EventType.CHAT_MESSAGE, on_chat_message)
    state["kick_monitor"].on_event(EventType.CHAT_SPIKE, on_chat_spike)

    # Registrar callbacks del pipeline
    pipeline.on_ready(on_clip_ready)
    pipeline.on_analyzed(on_clip_analyzed)  # Para actualizar resúmenes AI

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
    # Persistir en archivo
    config.save_channels(channels)

    # Si está corriendo, reiniciar el monitor para aplicar cambios
    if state["is_running"] and state["kick_monitor"]:
        await state["kick_monitor"].stop()
        await start_monitoring()

    return {"status": "updated", "channels": channels}


@app.post("/api/capture/{channel}")
async def manual_capture(channel: str, duration: int = 30):
    """
    Captura manual de un stream en vivo.

    Útil para:
    - Probar el sistema sin esperar un spike
    - Capturar momentos específicos manualmente
    """
    from recorders.clip_downloader import capture_live_stream

    await broadcast({"type": "log", "data": f"Iniciando captura manual de {channel} ({duration}s)..."})

    try:
        # Capturar stream
        clip_info = await capture_live_stream(
            channel=channel,
            output_dir=Path("./clips/raw"),
            duration=duration,
            platform="kick",
        )

        if clip_info and clip_info.file_path.exists():
            # Crear evento simulado para el pipeline
            from monitors.base_monitor import MonitorEvent, EventType

            event = MonitorEvent(
                event_type=EventType.CHAT_SPIKE,
                channel=channel,
                timestamp=datetime.now(),
                source="kick",
                data={
                    "velocity": 0,
                    "manual_capture": True,
                    "clip_path": str(clip_info.file_path),
                }
            )

            # Procesar en pipeline
            moment = await pipeline.handle_event(event)

            await broadcast({"type": "log", "data": f"Captura exitosa: {clip_info.file_path.name}"})

            return {
                "status": "success",
                "channel": channel,
                "duration": duration,
                "file": str(clip_info.file_path),
                "size_mb": clip_info.file_path.stat().st_size / 1024 / 1024,
            }
        else:
            await broadcast({"type": "log", "data": f"Error: No se pudo capturar {channel}"})
            return {"status": "error", "message": "Captura falló - el streamer puede no estar en vivo"}

    except Exception as e:
        await broadcast({"type": "log", "data": f"Error de captura: {str(e)}"})
        return {"status": "error", "message": str(e)}


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Obtiene el estado actual del pipeline y todos los momentos."""
    moments_data = []
    for m in pipeline.moments[:20]:  # Últimos 20
        moments_data.append({
            "id": m.id,
            "channel": m.channel,
            "status": m.status,
            "timestamp": m.timestamp.isoformat(),
            "clip_path": str(m.clip_path) if m.clip_path else None,
            "vertical_path": str(m.vertical_path) if m.vertical_path else None,
            "captioned_path": str(m.captioned_path) if m.captioned_path else None,
            "processed_path": str(m.processed_path) if m.processed_path else None,
            "error": m.error,
        })

    return {
        "moments": moments_data,
        "total": len(pipeline.moments),
        "processing_enabled": pipeline.auto_process,
        "has_reframer": pipeline.reframer is not None,
        "has_captioner": pipeline.captioner is not None,
    }


# Lista de streamers favoritos del usuario
MY_STREAMERS = [
    "esbebote", "marinagold", "elzeein", "sachauzumaki", "cristorata7", "kingteka", "milenkanolasco",
    "zullyy_cs", "benjaz", "neutroogg", "shuls_off",
    "diealis", "daarick", "noah_god", "luisormenoa27", "elsensei",
]


@app.get("/api/my-streamers")
async def get_my_streamers():
    """Obtiene el estado de MIS streamers favoritos (live u offline)."""
    from curl_cffi import requests as curl_requests
    import concurrent.futures

    def check_streamer(streamer):
        try:
            r = curl_requests.get(
                f"https://kick.com/api/v1/channels/{streamer}",
                impersonate="chrome",
                timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                livestream = data.get("livestream")
                is_live = livestream is not None
                return {
                    "slug": data.get("slug", streamer),
                    "username": data.get("user", {}).get("username", streamer),
                    "is_live": is_live,
                    "viewers": livestream.get("viewer_count", 0) if is_live and livestream else 0,
                    "title": livestream.get("session_title", "") if is_live and livestream else "",
                    "thumbnail": livestream.get("thumbnail", {}).get("url", "") if is_live and livestream else "",
                    "category": data.get("recent_categories", [{}])[0].get("name", "") if data.get("recent_categories") else "",
                }
            elif r.status_code == 404:
                return {"slug": streamer, "username": streamer, "is_live": False, "viewers": 0, "error": "not_found"}
        except Exception as e:
            return {"slug": streamer, "username": streamer, "is_live": False, "viewers": 0, "error": str(e)}
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(check_streamer, MY_STREAMERS))

    streamers = [r for r in results if r is not None]
    # Ordenar: live primero, luego por viewers
    streamers.sort(key=lambda x: (not x.get("is_live", False), -x.get("viewers", 0)))

    live_count = sum(1 for s in streamers if s.get("is_live"))
    return {"streamers": streamers, "total": len(streamers), "live_count": live_count}


@app.get("/api/live-streamers")
async def get_live_streamers():
    """Obtiene streamers EN VIVO de Kick - verifica lista de streamers conocidos."""
    from curl_cffi import requests as curl_requests

    live_streamers = []

    # Lista amplia de streamers a verificar (español + casino + populares)
    streamers_to_check = [
        # Español populares
        "esbebote", "elxokas", "juansguarnizo", "rivers_gg", "spreen", "carreraaa",
        "zormanworld", "westcol", "elmariana", "auronplay", "ibai",
        "rubius", "illojuan", "thegrefg", "djmariio", "ampeterby7",
        "byviruzz", "papigavi", "arigameplays", "juansguarnern", "fernanfloo",
        "luzu", "vegetta777", "alexby11", "staxx", "rickyedit",
        # Casino/Gambling
        "roshtein", "trainwreckstv", "xposed", "classybeef", "adinross",
        "stake", "foss", "ayezee", "yassuo", "prodigy", "stevewilldoit",
        # Inglés populares
        "xqc", "amouranth", "nickmercs", "clix", "ninja", "tfue",
        "mizkif", "hasanabi", "pokimane", "lirik", "summit1g", "shroud",
        # Más español
        "coscu", "manzana", "goncho", "brunenger", "zeko",
    ]

    # Usar curl_cffi que bypasea la protección de Kick
    def check_streamer(streamer):
        try:
            r = curl_requests.get(
                f"https://kick.com/api/v1/channels/{streamer}",
                impersonate="chrome",
                timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                livestream = data.get("livestream")
                if livestream is not None:
                    return {
                        "slug": data.get("slug", streamer),
                        "username": data.get("user", {}).get("username", streamer),
                        "viewers": livestream.get("viewer_count", 0),
                        "title": livestream.get("session_title", ""),
                        "thumbnail": livestream.get("thumbnail", {}).get("url", ""),
                        "category": data.get("recent_categories", [{}])[0].get("name", "") if data.get("recent_categories") else "",
                        "language": data.get("language", ""),
                    }
        except Exception:
            pass
        return None

    # Ejecutar en thread pool para no bloquear
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(check_streamer, streamers_to_check))

    live_streamers = [r for r in results if r is not None]

    # Ordenar por viewers
    live_streamers.sort(key=lambda x: x["viewers"], reverse=True)

    return {"live": live_streamers, "total": len(live_streamers)}


@app.post("/api/buffer/start/{channel}")
async def start_buffer(channel: str):
    """Inicia el buffer de stream para un canal (captura continua)."""
    if pipeline.multi_buffer:
        success = await pipeline.start_buffer_for_channel(channel, "kick")
        if success:
            await broadcast({"type": "log", "data": f"Buffer iniciado para {channel}"})
            return {"status": "started", "channel": channel}
        return {"status": "error", "message": "No se pudo iniciar el buffer"}
    return {"status": "error", "message": "Buffer no disponible"}


@app.post("/api/buffer/stop/{channel}")
async def stop_buffer(channel: str):
    """Detiene el buffer de stream para un canal."""
    if pipeline.multi_buffer:
        await pipeline.stop_buffer_for_channel(channel, "kick")
        await broadcast({"type": "log", "data": f"Buffer detenido para {channel}"})
        return {"status": "stopped", "channel": channel}
    return {"status": "error", "message": "Buffer no disponible"}


@app.get("/api/buffer/status")
async def get_buffer_status():
    """Obtiene el estado de todos los buffers activos."""
    if pipeline.multi_buffer:
        return {"status": "ok", "buffers": pipeline.multi_buffer.get_status()}
    return {"status": "disabled", "buffers": {}}


@app.post("/api/buffer/capture/{channel}")
async def capture_from_buffer(channel: str, pre_seconds: int = 10, post_seconds: int = 20):
    """
    Captura un clip del buffer incluyendo segundos ANTES del momento actual.

    Esto permite capturar el contexto que llevó al momento viral.
    """
    if not pipeline.multi_buffer:
        return {"status": "error", "message": "Buffer no disponible"}

    await broadcast({"type": "log", "data": f"Capturando desde buffer: {pre_seconds}s antes + {post_seconds}s después"})

    clip_path = await pipeline.capture_from_buffer(
        channel=channel,
        platform="kick",
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
    )

    if clip_path and clip_path.exists():
        await broadcast({"type": "log", "data": f"Clip capturado: {clip_path.name}"})

        # Crear momento para procesar
        from monitors.base_monitor import MonitorEvent, EventType
        event = MonitorEvent(
            event_type=EventType.CHAT_SPIKE,
            channel=channel,
            timestamp=datetime.now(),
            source="kick",
            data={
                "velocity": 0,
                "buffer_capture": True,
                "pre_seconds": pre_seconds,
                "post_seconds": post_seconds,
                "clip_path": str(clip_path),
            }
        )
        moment = await pipeline.handle_event(event)

        return {
            "status": "success",
            "channel": channel,
            "file": str(clip_path),
            "size_mb": clip_path.stat().st_size / 1024 / 1024,
            "moment_id": moment.id if moment else None,
        }

    return {"status": "error", "message": "No se pudo capturar desde el buffer"}


@app.post("/api/ai/analyze")
async def analyze_with_ai(transcript: str, channel: str = ""):
    """Analiza una transcripción con Gemini AI."""
    if not pipeline.gemini_analyzer:
        return {"status": "error", "message": "Gemini AI no disponible"}

    try:
        analysis = await pipeline.gemini_analyzer.analyze_transcript(
            transcript=transcript,
            channel=channel,
        )

        if analysis:
            return {
                "status": "success",
                "virality_score": analysis.virality_score,
                "is_viral": analysis.is_viral,
                "reasons": analysis.reasons,
                "suggested_title": analysis.suggested_title,
                "suggested_tags": analysis.suggested_tags,
                "content_type": analysis.content_type,
                "confidence": analysis.confidence,
            }
        return {"status": "error", "message": "Análisis falló"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================
# AI STREAM SUMMARIES
# ============================================

# Almacén de resúmenes por canal
stream_summaries: Dict[str, dict] = {}


@app.get("/api/ai/summaries")
async def get_ai_summaries():
    """Obtiene resúmenes AI de todos los streams monitoreados."""
    summaries = []

    for channel in state["channels"]:
        # Obtener velocidad actual del chat
        velocity = 0.0
        is_live = False
        if state["kick_monitor"]:
            velocity = state["kick_monitor"].get_chat_velocity(channel)
            if channel in state["kick_monitor"].channels:
                is_live = state["kick_monitor"].channels[channel].is_live

        # Obtener resumen existente o crear uno nuevo
        base_summary = stream_summaries.get(channel, {})

        # Crear copia con valores por defecto y datos en tiempo real
        summary = {
            "channel": channel,
            "clips_analyzed": base_summary.get("clips_analyzed", 0),
            "clips_viral": base_summary.get("clips_viral", 0),
            "clips_filtered": base_summary.get("clips_filtered", 0),
            "avg_score": base_summary.get("avg_score", 0.0),
            "recent_transcripts": base_summary.get("recent_transcripts", []),
            "highlights": base_summary.get("highlights", []),
            # Datos en tiempo real
            "velocity": round(velocity, 1),
            "is_live": is_live,
        }

        # Determinar mood y ai_summary basado en actividad
        if summary["clips_analyzed"] > 0:
            # Si hay clips analizados, usar los datos guardados
            summary["mood"] = base_summary.get("mood", "📺 Normal")
            summary["ai_summary"] = base_summary.get("ai_summary", "Monitoreando...")
        else:
            # Sin clips, mostrar estado basado en velocidad
            if velocity >= 30:
                summary["mood"] = "🔥 Alta actividad"
                summary["ai_summary"] = f"Chat muy activo ({velocity:.1f} msg/s)"
            elif velocity >= 10:
                summary["mood"] = "⚡ Activo"
                summary["ai_summary"] = f"Chat activo ({velocity:.1f} msg/s)"
            elif velocity > 0:
                summary["mood"] = "📺 Normal"
                summary["ai_summary"] = f"Chat tranquilo ({velocity:.1f} msg/s)"
            else:
                summary["mood"] = "😴 Sin actividad"
                summary["ai_summary"] = "Sin mensajes detectados"

        summaries.append(summary)

    return {"summaries": summaries}


@app.get("/api/ai/summary/{channel}")
async def get_channel_summary(channel: str):
    """Obtiene resumen AI detallado de un canal específico."""
    if channel not in stream_summaries:
        return {
            "channel": channel,
            "clips_analyzed": 0,
            "clips_viral": 0,
            "clips_filtered": 0,
            "avg_score": 0.0,
            "recent_transcripts": [],
            "ai_summary": "Sin actividad reciente",
            "mood": "neutral",
            "highlights": [],
        }
    return stream_summaries[channel]


async def update_stream_summary(channel: str, moment):
    """Actualiza el resumen de un stream después de analizar un clip."""
    if channel not in stream_summaries:
        stream_summaries[channel] = {
            "channel": channel,
            "clips_analyzed": 0,
            "clips_viral": 0,
            "clips_filtered": 0,
            "total_score": 0.0,
            "avg_score": 0.0,
            "recent_transcripts": [],
            "ai_summary": "",
            "mood": "neutral",
            "highlights": [],
            "last_update": None,
        }

    summary = stream_summaries[channel]
    summary["clips_analyzed"] += 1
    summary["total_score"] = summary.get("total_score", 0) + moment.ai_score
    summary["avg_score"] = summary["total_score"] / summary["clips_analyzed"]
    summary["last_update"] = datetime.now().isoformat()

    if moment.ai_score >= 0.6:
        summary["clips_viral"] += 1
        summary["highlights"].append({
            "timestamp": moment.timestamp.isoformat(),
            "score": moment.ai_score,
            "title": moment.ai_title,
            "tags": moment.ai_tags,
        })
        # Mantener solo últimos 10 highlights
        summary["highlights"] = summary["highlights"][-10:]
    else:
        summary["clips_filtered"] += 1

    # Determinar mood basado en actividad reciente
    if summary["clips_viral"] > 3 and summary["avg_score"] > 0.6:
        summary["mood"] = "🔥 En llamas"
    elif summary["clips_viral"] > 0:
        summary["mood"] = "⚡ Activo"
    elif summary["clips_analyzed"] > 5 and summary["avg_score"] < 0.3:
        summary["mood"] = "😴 Tranquilo"
    else:
        summary["mood"] = "📺 Normal"

    # Generar resumen AI si tenemos suficientes clips
    if summary["clips_analyzed"] % 5 == 0 and pipeline.gemini_analyzer:
        await generate_stream_summary_ai(channel, summary)

    # Broadcast update
    await broadcast({
        "type": "stream_summary",
        "data": summary
    })


async def generate_stream_summary_ai(channel: str, summary: dict):
    """Genera un resumen AI del stream."""
    if not pipeline.gemini_analyzer:
        return

    # Preparar contexto
    highlights_text = ""
    for h in summary.get("highlights", [])[-5:]:
        highlights_text += f"- {h.get('title', 'Sin título')} (score: {h.get('score', 0):.2f})\n"

    prompt = f"""Analiza este resumen de stream de {channel}:
- Clips analizados: {summary['clips_analyzed']}
- Clips virales: {summary['clips_viral']}
- Clips filtrados: {summary['clips_filtered']}
- Score promedio: {summary['avg_score']:.2f}

Highlights recientes:
{highlights_text if highlights_text else "Ninguno"}

Genera un resumen muy corto (1-2 líneas) del estado actual del stream.
Responde SOLO con el resumen, sin JSON."""

    try:
        import aiohttp
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={config.GEMINI_API_KEY}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 100}
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        summary["ai_summary"] = text.strip()
    except Exception as e:
        summary["ai_summary"] = f"Stream activo - {summary['clips_viral']} momentos virales detectados"


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
        "ai_enabled": pipeline.gemini_analyzer is not None,
        "buffer_enabled": pipeline.multi_buffer is not None,
        "ai_min_score": getattr(config, 'AI_MIN_VIRALITY_SCORE', 0.6),
        "buffer_pre_seconds": getattr(config, 'PRE_SPIKE_SECONDS', 10),
        "buffer_post_seconds": getattr(config, 'POST_SPIKE_SECONDS', 20),
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
    result = {"success": False, "error": "Platform not configured"}

    if platform == "tiktok":
        try:
            from publishers import TikTokUploader, TikTokConfig
            uploader = TikTokUploader(TikTokConfig(headless=True))
            if await uploader.login(wait_for_manual=False):
                result = await uploader.upload(
                    video_path=moment.processed_path or moment.clip_path,
                    caption=f"Momento viral de {moment.channel}",
                    tags=["fyp", "viral", "gaming", "win"],
                )
                if result.success:
                    moment.published_urls["tiktok"] = result.video_url
            await uploader.close()
        except Exception as e:
            result = {"success": False, "error": str(e)}

    elif platform == "youtube":
        try:
            from publishers import YouTubeShortsUploader
            uploader = YouTubeShortsUploader()
            if uploader.authenticate():
                result = await uploader.upload(
                    video_path=moment.processed_path or moment.clip_path,
                    title=f"Momento viral - {moment.channel}",
                    description=f"Capturado automáticamente",
                    tags=["shorts", "viral", "gaming"],
                )
                if result.success:
                    moment.published_urls["youtube"] = result.shorts_url
        except Exception as e:
            result = {"success": False, "error": str(e)}

    elif platform == "instagram":
        try:
            from publishers import InstagramReelsUploader, InstagramConfig
            uploader = InstagramReelsUploader()
            if await uploader.login():
                result = await uploader.upload_reel(
                    video_path=moment.processed_path or moment.clip_path,
                    caption=f"Momento viral de {moment.channel}",
                    hashtags=["reels", "viral", "gaming", "fyp"],
                )
                if result.success:
                    moment.published_urls["instagram"] = result.media_url
            uploader.logout()
        except Exception as e:
            result = {"success": False, "error": str(e)}

    return {"status": "ok" if getattr(result, 'success', False) else "error", "result": result}


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
    print("")
    print("=" * 50)
    print("  VIRAL CLIP MONITOR")
    print("=" * 50)
    print("")
    print("  Abriendo en: http://localhost:8000")
    print("")
    print("  Ctrl+C para detener")
    print("=" * 50)
    print("")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
