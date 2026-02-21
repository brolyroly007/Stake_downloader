"""Tests for the viral pipeline."""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from pipeline import ViralMoment, ViralPipeline
from monitors.base_monitor import MonitorEvent, EventType


class TestViralMoment:
    def test_moment_creation(self):
        moment = ViralMoment(
            id="test_001",
            event_type="spike",
            channel="testchannel",
            timestamp=datetime.now(),
            data={"velocity": 50},
        )
        assert moment.id == "test_001"
        assert moment.status == "detected"
        assert moment.clip_path is None
        assert moment.published_urls == {}
        assert moment.ai_score == 0.0

    def test_moment_defaults(self):
        moment = ViralMoment(
            id="test_002",
            event_type="spike",
            channel="test",
            timestamp=datetime.now(),
            data={},
        )
        assert moment.error is None
        assert moment.ai_title == ""
        assert moment.ai_tags == []
        assert moment.vertical_path is None
        assert moment.captioned_path is None
        assert moment.processed_path is None


class TestViralPipeline:
    @pytest.fixture
    def pipeline(self, tmp_path):
        with patch("pipeline.ClipDownloader"):
            with patch("pipeline.HAS_SMART_CLIPPER", False):
                with patch("pipeline.HAS_BUFFER", False):
                    with patch("pipeline.HAS_GEMINI", False):
                        with patch("pipeline.HAS_REFRAMER", False):
                            with patch("pipeline.HAS_CAPTIONS", False):
                                p = ViralPipeline(
                                    output_dir=tmp_path / "clips",
                                    auto_download=False,
                                    auto_process=False,
                                )
                                return p

    @pytest.mark.asyncio
    async def test_handle_event_creates_moment(self, pipeline):
        event = MonitorEvent(
            event_type=EventType.CHAT_SPIKE,
            source="kick",
            channel="testchannel",
            timestamp=datetime.now(),
            data={"velocity": 50},
        )
        moment = await pipeline.handle_event(event)
        assert moment is not None
        assert moment.channel == "testchannel"
        assert moment.status == "detected"
        assert len(pipeline.moments) == 1

    @pytest.mark.asyncio
    async def test_handle_event_limits_moments(self, pipeline):
        pipeline.max_moments = 3
        for i in range(5):
            event = MonitorEvent(
                event_type=EventType.CHAT_SPIKE,
                source="kick",
                channel=f"channel_{i}",
                timestamp=datetime.now(),
                data={},
            )
            await pipeline.handle_event(event)
        assert len(pipeline.moments) == 3

    @pytest.mark.asyncio
    async def test_get_clip_url_kick(self, pipeline):
        event = MonitorEvent(
            event_type=EventType.CHAT_SPIKE,
            source="kick",
            channel="testchannel",
            timestamp=datetime.now(),
            data={},
        )
        url = await pipeline._get_clip_url(event)
        assert url == "https://kick.com/testchannel"

    @pytest.mark.asyncio
    async def test_get_clip_url_unknown(self, pipeline):
        event = MonitorEvent(
            event_type=EventType.CHAT_SPIKE,
            source="unknown",
            channel="testchannel",
            timestamp=datetime.now(),
            data={},
        )
        url = await pipeline._get_clip_url(event)
        assert url is None

    def test_on_ready_callback(self, pipeline):
        callback = MagicMock()
        pipeline.on_ready(callback)
        assert callback in pipeline._on_moment_ready

    def test_on_analyzed_callback(self, pipeline):
        callback = MagicMock()
        pipeline.on_analyzed(callback)
        assert callback in pipeline._on_moment_analyzed

    def test_get_recent_moments(self, pipeline):
        moments = pipeline.get_recent_moments(limit=10)
        assert isinstance(moments, list)
        assert len(moments) == 0

    def test_get_stats(self, pipeline):
        stats = pipeline.get_stats()
        assert "total" in stats
        assert "ready" in stats
        assert "failed" in stats
        assert stats["total"] == 0

    def test_get_stats_with_moments(self, pipeline):
        pipeline.moments.append(
            ViralMoment(
                id="m1",
                event_type="spike",
                channel="test",
                timestamp=datetime.now(),
                data={},
                status="ready",
            )
        )
        pipeline.moments.append(
            ViralMoment(
                id="m2",
                event_type="spike",
                channel="test",
                timestamp=datetime.now(),
                data={},
                status="failed",
            )
        )
        stats = pipeline.get_stats()
        assert stats["total"] == 2
        assert stats["ready"] == 1
        assert stats["failed"] == 1
