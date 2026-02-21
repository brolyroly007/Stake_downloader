"""Tests for the FastAPI application endpoints."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def client():
    """Create a test client."""
    # Patch monitors before importing app
    with patch("app.KickMonitor"):
        with patch("app.ViralPipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.moments = []
            mock_pipeline.get_recent_moments.return_value = []
            mock_pipeline.get_stats.return_value = {"total": 0}
            mock_pipeline.gemini_analyzer = None
            mock_pipeline.multi_buffer = None
            mock_pipeline.auto_process = True
            mock_pipeline.reframer = None
            mock_pipeline.captioner = None
            MockPipeline.return_value = mock_pipeline

            from fastapi.testclient import TestClient

            # Need to reload app module with mocks
            import importlib
            import app as app_module

            importlib.reload(app_module)
            yield TestClient(app_module.app)


class TestStatusEndpoint:
    def test_get_status(self, client):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "is_running" in data
        assert "stats" in data
        assert "events" in data


class TestClipsEndpoint:
    def test_get_clips(self, client):
        response = client.get("/api/clips")
        assert response.status_code == 200
        data = response.json()
        assert "clips" in data
        assert "stats" in data


class TestDashboard:
    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
