"""
Clase base abstracta para todos los monitores.
Define la interfaz común que deben implementar todos los monitores.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio


class EventType(Enum):
    """Tipos de eventos que pueden ser detectados."""
    CHAT_MESSAGE = "chat_message"
    CHAT_SPIKE = "chat_spike"  # Pico de mensajes
    BIG_WIN = "big_win"
    DONATION = "donation"
    RAID = "raid"
    SUBSCRIPTION = "subscription"
    STREAM_START = "stream_start"
    STREAM_END = "stream_end"
    CLIP_CREATED = "clip_created"
    CUSTOM = "custom"


@dataclass
class MonitorEvent:
    """Estructura de datos para eventos detectados."""
    event_type: EventType
    source: str  # Plataforma (kick, twitch, stake)
    channel: str  # Canal o usuario
    timestamp: datetime
    data: Dict[str, Any]
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        """Genera un ID único para el evento."""
        return f"{self.source}_{self.channel}_{self.event_type.value}_{self.timestamp.timestamp()}"


class BaseMonitor(ABC):
    """
    Clase base abstracta para monitores de plataformas.

    Todos los monitores deben heredar de esta clase e implementar
    los métodos abstractos.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.is_connected = False
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._stats = {
            "events_received": 0,
            "events_processed": 0,
            "errors": 0,
            "reconnections": 0,
            "started_at": None,
            "last_event_at": None,
        }

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establece conexión con la plataforma.
        Debe retornar True si la conexión fue exitosa.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Cierra la conexión con la plataforma."""
        pass

    @abstractmethod
    async def _listen(self) -> None:
        """
        Loop principal que escucha eventos de la plataforma.
        Debe llamar a _emit_event cuando detecte un evento relevante.
        """
        pass

    async def start(self) -> None:
        """Inicia el monitor."""
        if self.is_running:
            return

        self.is_running = True
        self._stats["started_at"] = datetime.now()

        # Conectar
        success = await self.connect()
        if not success:
            self.is_running = False
            raise ConnectionError(f"No se pudo conectar al monitor {self.name}")

        # Iniciar loop de escucha en background
        asyncio.create_task(self._run_listen_loop())

    async def stop(self) -> None:
        """Detiene el monitor."""
        self.is_running = False
        await self.disconnect()

    async def _run_listen_loop(self) -> None:
        """Ejecuta el loop de escucha con manejo de reconexiones."""
        while self.is_running:
            try:
                await self._listen()
            except Exception as e:
                self._stats["errors"] += 1
                if self.is_running:
                    # Intentar reconectar
                    self._stats["reconnections"] += 1
                    await asyncio.sleep(5)  # Esperar antes de reconectar
                    try:
                        await self.connect()
                    except Exception:
                        pass

    def on_event(self, event_type: EventType, handler: Callable) -> None:
        """
        Registra un handler para un tipo de evento específico.

        Args:
            event_type: Tipo de evento a escuchar
            handler: Función async que será llamada con el evento
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def on_any_event(self, handler: Callable) -> None:
        """Registra un handler para todos los eventos."""
        for event_type in EventType:
            self.on_event(event_type, handler)

    async def _emit_event(self, event: MonitorEvent) -> None:
        """
        Emite un evento a todos los handlers registrados.

        Args:
            event: Evento a emitir
        """
        self._stats["events_received"] += 1
        self._stats["last_event_at"] = datetime.now()

        # Añadir a la cola
        await self._event_queue.put(event)

        # Llamar a handlers
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                self._stats["events_processed"] += 1
            except Exception as e:
                self._stats["errors"] += 1

    async def get_next_event(self, timeout: Optional[float] = None) -> Optional[MonitorEvent]:
        """
        Obtiene el siguiente evento de la cola.

        Args:
            timeout: Tiempo máximo de espera en segundos

        Returns:
            MonitorEvent o None si hay timeout
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=timeout
                )
            return await self._event_queue.get()
        except asyncio.TimeoutError:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del monitor."""
        return {
            **self._stats,
            "name": self.name,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "queue_size": self._event_queue.qsize(),
        }
