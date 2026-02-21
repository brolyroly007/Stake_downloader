"""
Monitor de Kick.com para detectar eventos en streams.

Utiliza WebSockets para recibir eventos en tiempo real:
- Mensajes de chat
- Picos de actividad en chat
- Clips creados
- Eventos de stream (inicio, fin, raids, etc.)
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import websockets
from websockets.client import WebSocketClientProtocol

from .base_monitor import BaseMonitor, MonitorEvent, EventType

# Intentar importar KickApi, si no está disponible usar fallback
try:
    from kickapi import KickAPI
    HAS_KICKAPI = True
except ImportError:
    HAS_KICKAPI = False


@dataclass
class KickChannel:
    """Información de un canal de Kick."""
    slug: str  # username/slug del canal
    channel_id: Optional[int] = None
    chatroom_id: Optional[int] = None
    is_live: bool = False
    viewer_count: int = 0
    stream_title: Optional[str] = None


class ChatVelocityTracker:
    """
    Rastrea la velocidad de mensajes en el chat.
    Detecta picos de actividad que pueden indicar momentos virales.

    Usa un algoritmo de detección de picos reales:
    - Compara velocidad actual vs promedio reciente
    - Solo dispara cuando hay un INCREMENTO significativo (>2x el promedio)
    - Cooldown más largo para evitar falsos positivos
    """

    def __init__(self, window_seconds: int = 10, spike_threshold: int = 50):
        self.window_seconds = window_seconds
        self.spike_threshold = spike_threshold  # Velocidad mínima para considerar spike
        self._messages: List[datetime] = []
        self._last_spike_time: Optional[datetime] = None
        self._cooldown_seconds = 60  # Cooldown más largo (1 minuto)

        # Para detección de picos reales (algoritmo divide y vencerás)
        self._velocity_history: List[float] = []  # Historial de velocidades
        self._history_max_size = 30  # Últimos 30 samples (5 min aprox)
        self._spike_multiplier = 2.0  # Velocidad debe ser 2x el promedio

    def add_message(self) -> bool:
        """
        Registra un mensaje y retorna True si hay un spike REAL.

        Un spike real requiere:
        1. Velocidad >= threshold mínimo
        2. Velocidad >= 2x el promedio histórico (pico real, no actividad constante)
        3. No estar en cooldown
        """
        now = datetime.now()
        self._messages.append(now)

        # Limpiar mensajes fuera de la ventana
        cutoff = now.timestamp() - self.window_seconds
        self._messages = [
            msg for msg in self._messages
            if msg.timestamp() > cutoff
        ]

        # Calcular velocidad actual
        velocity = len(self._messages) / self.window_seconds

        # Actualizar historial de velocidades (cada ~10 mensajes)
        if len(self._messages) % 10 == 0 and velocity > 0:
            self._velocity_history.append(velocity)
            if len(self._velocity_history) > self._history_max_size:
                self._velocity_history = self._velocity_history[-self._history_max_size:]

        # Verificar cooldown
        if self._last_spike_time:
            time_since_spike = (now - self._last_spike_time).total_seconds()
            if time_since_spike < self._cooldown_seconds:
                return False

        # Verificar si es un spike REAL
        if velocity >= self.spike_threshold:
            # Calcular promedio histórico (excluyendo el valor actual)
            if len(self._velocity_history) >= 3:
                avg_velocity = sum(self._velocity_history[:-1]) / len(self._velocity_history[:-1])

                # Solo es spike si es significativamente mayor al promedio
                if velocity >= avg_velocity * self._spike_multiplier:
                    self._last_spike_time = now
                    return True
            else:
                # No hay suficiente historial, usar threshold absoluto
                self._last_spike_time = now
                return True

        return False

    def get_velocity(self) -> float:
        """Retorna mensajes por segundo actual."""
        now = datetime.now()
        cutoff = now.timestamp() - self.window_seconds
        recent = [msg for msg in self._messages if msg.timestamp() > cutoff]
        return len(recent) / self.window_seconds


class KickMonitor(BaseMonitor):
    """
    Monitor para streams de Kick.com.

    Características:
    - Conexión via WebSocket (Pusher)
    - Detección de picos de chat
    - Eventos de clips
    - Información de stream en tiempo real
    """

    # URL base del WebSocket de Kick (usa Pusher)
    # Key actualizada: 32cbd69e4b950bf97679
    PUSHER_WS_URL = "wss://ws-us2.pusher.com/app/32cbd69e4b950bf97679"
    PUSHER_PARAMS = "?protocol=7&client=js&version=7.4.0&flash=false"

    def __init__(
        self,
        channels: List[str],
        chat_velocity_threshold: int = 50,
        detect_chat_spikes: bool = True,
    ):
        """
        Args:
            channels: Lista de slugs de canales a monitorear
            chat_velocity_threshold: Mensajes/segundo para considerar spike
            detect_chat_spikes: Si debe detectar picos de chat
        """
        super().__init__(name="KickMonitor")

        self.channels: Dict[str, KickChannel] = {
            slug: KickChannel(slug=slug) for slug in channels
        }
        self.detect_chat_spikes = detect_chat_spikes

        # Trackers de velocidad de chat por canal
        self._velocity_trackers: Dict[str, ChatVelocityTracker] = {
            slug: ChatVelocityTracker(spike_threshold=chat_velocity_threshold)
            for slug in channels
        }

        self._ws: Optional[WebSocketClientProtocol] = None
        self._kickapi: Optional[Any] = None

        if HAS_KICKAPI:
            self._kickapi = KickAPI()

    async def connect(self) -> bool:
        """Establece conexión con el WebSocket de Kick."""
        try:
            # Conectar al WebSocket de Pusher
            ws_url = f"{self.PUSHER_WS_URL}{self.PUSHER_PARAMS}"
            self._ws = await websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
            )

            # Esperar mensaje de conexión
            response = await self._ws.recv()
            data = json.loads(response)

            if data.get("event") == "pusher:connection_established":
                self.is_connected = True

                # Suscribirse a los canales
                for channel in self.channels.values():
                    await self._subscribe_to_channel(channel)

                return True

            return False

        except Exception as e:
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Cierra la conexión WebSocket."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self.is_connected = False

    def get_chat_velocity(self, channel_slug: str) -> float:
        """Obtiene la velocidad de chat actual de un canal."""
        if channel_slug in self._velocity_trackers:
            return self._velocity_trackers[channel_slug].get_velocity()
        return 0.0

    def get_all_velocities(self) -> Dict[str, float]:
        """Obtiene las velocidades de todos los canales."""
        return {
            slug: tracker.get_velocity()
            for slug, tracker in self._velocity_trackers.items()
        }

    async def _fetch_channel_info(self, channel: KickChannel) -> None:
        """Obtiene información del canal via API HTTP."""
        try:
            # Intentar con curl_cffi primero (bypass Cloudflare)
            try:
                from curl_cffi.requests import AsyncSession
                async with AsyncSession(impersonate="chrome") as session:
                    url = f"https://kick.com/api/v2/channels/{channel.slug}"
                    response = await session.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        channel.channel_id = data.get("id")
                        channel.chatroom_id = data.get("chatroom", {}).get("id")
                        channel.is_live = data.get("livestream") is not None
                        if channel.is_live:
                            channel.viewer_count = data.get("livestream", {}).get("viewer_count", 0)
                            channel.stream_title = data.get("livestream", {}).get("session_title")
                        return
            except ImportError:
                pass

            # Fallback a KickApi si está disponible
            if self._kickapi:
                channel_info = await asyncio.to_thread(
                    self._kickapi.get_channel,
                    channel.slug
                )
                if channel_info:
                    channel.channel_id = channel_info.get("id")
                    channel.chatroom_id = channel_info.get("chatroom", {}).get("id")
                    channel.is_live = channel_info.get("livestream") is not None

        except Exception:
            pass

    async def _subscribe_to_channel(self, channel: KickChannel) -> None:
        """Suscribe al chatroom de un canal."""
        if not self._ws:
            return

        # Obtener chatroom_id si no lo tenemos
        if not channel.chatroom_id:
            await self._fetch_channel_info(channel)

        # Suscribirse al canal de chat
        # Formato: chatrooms.{chatroom_id}.v2 (versión 2 del protocolo)
        chatroom_channel = f"chatrooms.{channel.chatroom_id or channel.slug}.v2"

        subscribe_message = json.dumps({
            "event": "pusher:subscribe",
            "data": {
                "channel": chatroom_channel
            }
        })

        await self._ws.send(subscribe_message)

        # También suscribirse al canal general para eventos de stream
        channel_channel = f"channel.{channel.slug}"
        subscribe_message = json.dumps({
            "event": "pusher:subscribe",
            "data": {
                "channel": channel_channel
            }
        })

        await self._ws.send(subscribe_message)

    async def _listen(self) -> None:
        """Loop principal de escucha de eventos."""
        if not self._ws:
            return

        try:
            async for message in self._ws:
                await self._handle_message(message)
        except websockets.ConnectionClosed:
            self.is_connected = False
            raise

    async def _handle_message(self, message: str) -> None:
        """Procesa un mensaje recibido del WebSocket."""
        try:
            data = json.loads(message)
            event_name = data.get("event", "")
            channel_name = data.get("channel", "")

            # Ignorar eventos de sistema de Pusher
            if event_name.startswith("pusher:"):
                return

            # Parsear data interna
            event_data = {}
            if "data" in data:
                try:
                    event_data = json.loads(data["data"])
                except (json.JSONDecodeError, TypeError):
                    event_data = data.get("data", {})

            # Extraer slug del canal
            channel_slug = self._extract_channel_slug(channel_name)

            # Procesar según tipo de evento
            if event_name == "App\\Events\\ChatMessageEvent":
                await self._handle_chat_message(channel_slug, event_data)

            elif event_name == "App\\Events\\StreamerIsLive":
                await self._handle_stream_start(channel_slug, event_data)

            elif event_name == "App\\Events\\StopStreamBroadcast":
                await self._handle_stream_end(channel_slug, event_data)

            elif event_name == "App\\Events\\FollowersUpdated":
                # Nuevo follow - podría indicar un momento viral
                pass

            elif event_name == "App\\Events\\SubscriptionEvent":
                await self._handle_subscription(channel_slug, event_data)

        except json.JSONDecodeError:
            pass

    def _extract_channel_slug(self, channel_name: str) -> str:
        """Extrae el slug del canal del nombre del canal de Pusher."""
        # Formatos posibles:
        # - chatrooms.12345.v2
        # - channel.username
        parts = channel_name.split(".")
        if len(parts) >= 2:
            chatroom_id_str = parts[1]
            # Buscar el slug que corresponde a este chatroom_id
            try:
                chatroom_id = int(chatroom_id_str)
                for slug, channel in self.channels.items():
                    if channel.chatroom_id == chatroom_id:
                        return slug
            except ValueError:
                # No es un número, probablemente es un username
                return chatroom_id_str
            return chatroom_id_str
        return channel_name

    async def _handle_chat_message(
        self,
        channel_slug: str,
        data: Dict[str, Any]
    ) -> None:
        """Procesa un mensaje de chat."""
        # Crear evento de mensaje
        event = MonitorEvent(
            event_type=EventType.CHAT_MESSAGE,
            source="kick",
            channel=channel_slug,
            timestamp=datetime.now(),
            data={
                "username": data.get("sender", {}).get("username", ""),
                "message": data.get("content", ""),
                "user_id": data.get("sender", {}).get("id"),
                "is_subscriber": data.get("sender", {}).get("is_subscriber", False),
                "badges": data.get("sender", {}).get("identity", {}).get("badges", []),
            },
            raw_data=data,
        )

        await self._emit_event(event)

        # Verificar spike de chat
        if self.detect_chat_spikes and channel_slug in self._velocity_trackers:
            tracker = self._velocity_trackers[channel_slug]
            is_spike = tracker.add_message()

            if is_spike:
                spike_event = MonitorEvent(
                    event_type=EventType.CHAT_SPIKE,
                    source="kick",
                    channel=channel_slug,
                    timestamp=datetime.now(),
                    data={
                        "velocity": tracker.get_velocity(),
                        "threshold": tracker.spike_threshold,
                        "trigger_message": data.get("content", ""),
                    },
                    metadata={
                        "virality_signal": True,
                        "signal_weight": 0.3,
                    }
                )
                await self._emit_event(spike_event)

    async def _handle_stream_start(
        self,
        channel_slug: str,
        data: Dict[str, Any]
    ) -> None:
        """Procesa evento de inicio de stream."""
        if channel_slug in self.channels:
            self.channels[channel_slug].is_live = True

        event = MonitorEvent(
            event_type=EventType.STREAM_START,
            source="kick",
            channel=channel_slug,
            timestamp=datetime.now(),
            data={
                "stream_title": data.get("livestream", {}).get("session_title", ""),
                "viewer_count": data.get("livestream", {}).get("viewers", 0),
            },
            raw_data=data,
        )

        await self._emit_event(event)

    async def _handle_stream_end(
        self,
        channel_slug: str,
        data: Dict[str, Any]
    ) -> None:
        """Procesa evento de fin de stream."""
        if channel_slug in self.channels:
            self.channels[channel_slug].is_live = False

        event = MonitorEvent(
            event_type=EventType.STREAM_END,
            source="kick",
            channel=channel_slug,
            timestamp=datetime.now(),
            data=data,
            raw_data=data,
        )

        await self._emit_event(event)

    async def _handle_subscription(
        self,
        channel_slug: str,
        data: Dict[str, Any]
    ) -> None:
        """Procesa evento de suscripción."""
        event = MonitorEvent(
            event_type=EventType.SUBSCRIPTION,
            source="kick",
            channel=channel_slug,
            timestamp=datetime.now(),
            data={
                "username": data.get("username", ""),
                "months": data.get("months", 1),
            },
            raw_data=data,
        )

        await self._emit_event(event)
async def main():
    """Ejemplo de uso del KickMonitor."""
    from core.logger import logger

    # Crear monitor para algunos canales
    channels = ["trainwreckstv", "roshtein", "xposed"]
    monitor = KickMonitor(channels=channels, chat_velocity_threshold=50)

    # Registrar handlers
    async def on_chat_spike(event: MonitorEvent):
        logger.info(
            f"🔥 CHAT SPIKE en {event.channel}! "
            f"Velocidad: {event.data['velocity']:.1f} msg/s"
        )

    async def on_chat_message(event: MonitorEvent):
        logger.debug(
            f"[{event.channel}] {event.data['username']}: {event.data['message'][:50]}"
        )

    monitor.on_event(EventType.CHAT_SPIKE, on_chat_spike)
    monitor.on_event(EventType.CHAT_MESSAGE, on_chat_message)

    try:
        await monitor.start()
        logger.info(f"Monitor iniciado para canales: {channels}")

        # Mantener corriendo
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Deteniendo monitor...")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
