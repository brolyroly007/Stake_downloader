"""Tests for the monitor system."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from monitors.base_monitor import BaseMonitor, EventType, MonitorEvent


class ConcreteMonitor(BaseMonitor):
    """Concrete implementation for testing."""

    def __init__(self):
        super().__init__("test_monitor")
        self.connect_called = False
        self.disconnect_called = False

    async def connect(self) -> bool:
        self.connect_called = True
        self.is_connected = True
        return True

    async def disconnect(self) -> None:
        self.disconnect_called = True
        self.is_connected = False

    async def _listen(self) -> None:
        # Simulate listening then stopping
        await asyncio.sleep(0.1)
        self.is_running = False


@pytest.fixture
def monitor():
    return ConcreteMonitor()


class TestMonitorEvent:
    def test_event_creation(self):
        event = MonitorEvent(
            event_type=EventType.CHAT_MESSAGE,
            source="kick",
            channel="testchannel",
            timestamp=datetime.now(),
            data={"message": "hello"},
        )
        assert event.event_type == EventType.CHAT_MESSAGE
        assert event.source == "kick"
        assert event.channel == "testchannel"

    def test_event_id_generation(self):
        event = MonitorEvent(
            event_type=EventType.CHAT_SPIKE,
            source="kick",
            channel="testchannel",
            timestamp=datetime.now(),
            data={},
        )
        event_id = event.event_id
        assert "kick" in event_id
        assert "testchannel" in event_id
        assert "chat_spike" in event_id

    def test_event_metadata_default(self):
        event = MonitorEvent(
            event_type=EventType.CHAT_MESSAGE,
            source="kick",
            channel="test",
            timestamp=datetime.now(),
            data={},
        )
        assert event.metadata == {}


class TestEventType:
    def test_all_event_types_exist(self):
        assert EventType.CHAT_MESSAGE.value == "chat_message"
        assert EventType.CHAT_SPIKE.value == "chat_spike"
        assert EventType.BIG_WIN.value == "big_win"
        assert EventType.STREAM_START.value == "stream_start"
        assert EventType.STREAM_END.value == "stream_end"
        assert EventType.CLIP_CREATED.value == "clip_created"


class TestBaseMonitor:
    def test_init(self, monitor):
        assert monitor.name == "test_monitor"
        assert monitor.is_running is False
        assert monitor.is_connected is False

    def test_register_event_handler(self, monitor):
        handler = MagicMock()
        monitor.on_event(EventType.CHAT_MESSAGE, handler)
        assert EventType.CHAT_MESSAGE in monitor._event_handlers
        assert handler in monitor._event_handlers[EventType.CHAT_MESSAGE]

    def test_register_multiple_handlers(self, monitor):
        handler1 = MagicMock()
        handler2 = MagicMock()
        monitor.on_event(EventType.CHAT_MESSAGE, handler1)
        monitor.on_event(EventType.CHAT_MESSAGE, handler2)
        assert len(monitor._event_handlers[EventType.CHAT_MESSAGE]) == 2

    @pytest.mark.asyncio
    async def test_emit_event(self, monitor):
        handler = AsyncMock()
        monitor.on_event(EventType.CHAT_SPIKE, handler)

        event = MonitorEvent(
            event_type=EventType.CHAT_SPIKE,
            source="kick",
            channel="test",
            timestamp=datetime.now(),
            data={"velocity": 50},
        )
        await monitor._emit_event(event)
        handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_emit_event_sync_handler(self, monitor):
        handler = MagicMock()
        monitor.on_event(EventType.CHAT_MESSAGE, handler)

        event = MonitorEvent(
            event_type=EventType.CHAT_MESSAGE,
            source="kick",
            channel="test",
            timestamp=datetime.now(),
            data={},
        )
        await monitor._emit_event(event)
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_event_updates_stats(self, monitor):
        event = MonitorEvent(
            event_type=EventType.CHAT_MESSAGE,
            source="kick",
            channel="test",
            timestamp=datetime.now(),
            data={},
        )
        await monitor._emit_event(event)
        assert monitor._stats["events_received"] == 1

    def test_get_stats(self, monitor):
        stats = monitor.get_stats()
        assert "name" in stats
        assert "is_running" in stats
        assert "is_connected" in stats
        assert "events_received" in stats
        assert stats["name"] == "test_monitor"

    @pytest.mark.asyncio
    async def test_stop(self, monitor):
        monitor.is_running = True
        await monitor.stop()
        assert monitor.is_running is False
        assert monitor.disconnect_called is True

    @pytest.mark.asyncio
    async def test_get_next_event_timeout(self, monitor):
        result = await monitor.get_next_event(timeout=0.1)
        assert result is None
