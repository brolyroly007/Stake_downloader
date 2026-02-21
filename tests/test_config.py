"""Tests for configuration modules."""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch


class TestRootConfig:
    def test_load_channels_default(self):
        """When no channels file exists, returns defaults."""
        with patch("config.CHANNELS_FILE", Path("/nonexistent/file.json")):
            import config

            channels = config.load_channels()
            assert isinstance(channels, list)
            assert len(channels) > 0
            assert "trainwreckstv" in channels

    def test_save_and_load_channels(self, tmp_path):
        """Channels can be saved and loaded."""
        import config

        channels_file = tmp_path / "channels.json"
        config.CHANNELS_FILE = channels_file

        test_channels = ["channel1", "channel2", "channel3"]
        result = config.save_channels(test_channels)
        assert result is True
        assert channels_file.exists()

        loaded = config.load_channels()
        assert loaded == test_channels

    def test_config_thresholds(self):
        """Config thresholds are reasonable values."""
        import config

        assert 0 < config.CHAT_VELOCITY_THRESHOLD < 1000
        assert 0.0 <= config.VIRALITY_THRESHOLD <= 1.0

    def test_gemini_key_not_hardcoded(self):
        """Gemini API key should not be hardcoded."""
        import config

        # Should be empty or from env, never a real key
        if config.GEMINI_API_KEY:
            assert config.GEMINI_API_KEY == os.getenv("GEMINI_API_KEY", "")


class TestCoreConfig:
    def test_settings_defaults(self):
        """Settings have reasonable defaults."""
        from core.config import Settings

        settings = Settings()
        assert settings.debug is True
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert settings.virality_score_threshold >= 0.0
        assert settings.virality_score_threshold <= 1.0

    def test_kick_channels_list(self):
        """Kick channels string parses to list."""
        from core.config import Settings

        settings = Settings(kick_channels="channel1,channel2,channel3")
        assert settings.kick_channels_list == ["channel1", "channel2", "channel3"]

    def test_kick_channels_list_with_spaces(self):
        """Kick channels handles whitespace."""
        from core.config import Settings

        settings = Settings(kick_channels="channel1, channel2 , channel3")
        assert settings.kick_channels_list == ["channel1", "channel2", "channel3"]

    def test_log_level_validation(self):
        """Invalid log level raises error."""
        from core.config import Settings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")

    def test_log_level_case_insensitive(self):
        """Log level accepts any case."""
        from core.config import Settings

        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_whisper_model_validation(self):
        """Invalid whisper model raises error."""
        from core.config import Settings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(whisper_model="invalid_model")

    def test_virality_threshold_bounds(self):
        """Virality threshold must be between 0 and 1."""
        from core.config import Settings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(virality_score_threshold=1.5)

        with pytest.raises(ValidationError):
            Settings(virality_score_threshold=-0.1)

    def test_ensure_directories(self, tmp_path):
        """ensure_directories creates required dirs."""
        from core.config import Settings

        settings = Settings(
            output_dir=tmp_path / "output",
            clips_dir=tmp_path / "clips",
            temp_dir=tmp_path / "temp",
        )
        settings.ensure_directories()

        assert (tmp_path / "output").exists()
        assert (tmp_path / "clips").exists()
        assert (tmp_path / "temp").exists()

    def test_proxy_list_no_file(self):
        """Proxy list returns empty when no file exists."""
        from core.config import Settings

        settings = Settings(use_proxies=True, proxy_list_file=Path("/nonexistent.txt"))
        assert settings.get_proxy_list() == []

    def test_proxy_list_disabled(self):
        """Proxy list returns empty when disabled."""
        from core.config import Settings

        settings = Settings(use_proxies=False)
        assert settings.get_proxy_list() == []
