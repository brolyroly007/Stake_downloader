"""Tests for the virality scoring system."""

import pytest
from datetime import datetime
from detectors.virality_scorer import (
    ViralityScorer,
    ViralitySignal,
    SignalType,
)


@pytest.fixture
def scorer():
    """Create a scorer with default settings."""
    return ViralityScorer(threshold=0.7, cooldown_seconds=5.0)


@pytest.fixture
def low_threshold_scorer():
    """Create a scorer with low threshold for easier triggering."""
    return ViralityScorer(threshold=0.3, cooldown_seconds=1.0)


class TestViralityScorer:
    def test_init_defaults(self, scorer):
        assert scorer.threshold == 0.7
        assert scorer.cooldown == 5.0
        assert scorer.signal_window == 30.0

    def test_custom_weights(self):
        custom_weights = {SignalType.CHAT_VELOCITY: 0.5}
        scorer = ViralityScorer(weights=custom_weights)
        assert scorer.weights[SignalType.CHAT_VELOCITY] == 0.5
        # Other weights should still be defaults
        assert scorer.weights[SignalType.BIG_WIN] == 0.25

    def test_add_signal_below_threshold(self, scorer):
        signal = ViralitySignal(
            signal_type=SignalType.CHAT_VELOCITY,
            value=0.1,
            weight=0.3,
            timestamp=datetime.now(),
            source="kick",
            channel="testchannel",
        )
        result = scorer.add_signal(signal)
        assert result is None  # Below threshold

    def test_add_signal_above_threshold(self, low_threshold_scorer):
        signal = ViralitySignal(
            signal_type=SignalType.CHAT_VELOCITY,
            value=0.95,
            weight=0.3,
            timestamp=datetime.now(),
            source="kick",
            channel="testchannel",
        )
        result = low_threshold_scorer.add_signal(signal)
        # With a single signal at 0.95 * 0.3 = 0.285, just below 0.3
        # Need to add another signal to push above threshold
        signal2 = ViralitySignal(
            signal_type=SignalType.BIG_WIN,
            value=0.9,
            weight=0.25,
            timestamp=datetime.now(),
            source="stake",
            channel="testchannel",
        )
        result = low_threshold_scorer.add_signal(signal2)
        assert result is not None
        assert result.is_viral is True
        assert result.score > 0.3

    def test_normalize_value_big_win(self, scorer):
        # $50,000 win should normalize to 0.5 (range 0-100k)
        normalized = scorer.normalize_value(SignalType.BIG_WIN, 50000)
        assert normalized == pytest.approx(0.5)

    def test_normalize_value_clamps(self, scorer):
        # Above max should clamp to 1.0
        normalized = scorer.normalize_value(SignalType.BIG_WIN, 200000)
        assert normalized == 1.0

        # Below min should clamp to 0.0
        normalized = scorer.normalize_value(SignalType.BIG_WIN, -100)
        assert normalized == 0.0

    def test_normalize_audio_peak(self, scorer):
        # -30 dB is middle range between -60 and 0
        normalized = scorer.normalize_value(SignalType.AUDIO_PEAK, -30)
        assert normalized == pytest.approx(0.5)

    def test_normalize_unknown_type(self, scorer):
        # Unknown type should use 0-100 range
        normalized = scorer.normalize_value(SignalType.CUSTOM, 50)
        assert normalized == pytest.approx(0.5)

    def test_create_signal(self, scorer):
        signal = scorer.create_signal(
            signal_type=SignalType.BIG_WIN,
            raw_value=75000,
            source="stake",
            channel="roshtein",
            metadata={"game": "slots"},
        )
        assert signal.signal_type == SignalType.BIG_WIN
        assert signal.raw_value == 75000
        assert signal.value == pytest.approx(0.75)
        assert signal.source == "stake"
        assert signal.channel == "roshtein"
        assert signal.metadata == {"game": "slots"}

    def test_cooldown_prevents_duplicate_triggers(self):
        # Use very low threshold so a single strong signal triggers
        scorer = ViralityScorer(threshold=0.1, cooldown_seconds=5.0)

        # First trigger - BIG_WIN value=1.0 * weight=0.25 = 0.25 > 0.1
        signal1 = ViralitySignal(
            signal_type=SignalType.BIG_WIN,
            value=1.0,
            weight=0.25,
            timestamp=datetime.now(),
            source="stake",
            channel="testchannel",
        )
        result1 = scorer.add_signal(signal1)
        assert result1 is not None

        # Second trigger should be blocked by cooldown
        signal2 = ViralitySignal(
            signal_type=SignalType.CHAT_VELOCITY,
            value=1.0,
            weight=0.3,
            timestamp=datetime.now(),
            source="kick",
            channel="testchannel",
        )
        result2 = scorer.add_signal(signal2)
        assert result2 is None  # Blocked by cooldown

    def test_get_current_score(self, scorer):
        score = scorer.get_current_score("nonexistent")
        assert score.score == 0.0
        assert score.is_viral is False

    def test_get_active_channels(self, scorer):
        signal = ViralitySignal(
            signal_type=SignalType.CHAT_VELOCITY,
            value=0.5,
            weight=0.3,
            timestamp=datetime.now(),
            source="kick",
            channel="testchannel",
        )
        scorer.add_signal(signal)
        assert "testchannel" in scorer.get_active_channels()

    def test_clear_channel(self, scorer):
        signal = ViralitySignal(
            signal_type=SignalType.CHAT_VELOCITY,
            value=0.5,
            weight=0.3,
            timestamp=datetime.now(),
            source="kick",
            channel="testchannel",
        )
        scorer.add_signal(signal)
        scorer.clear_channel("testchannel")
        assert "testchannel" not in scorer.get_active_channels()

    def test_get_stats(self, scorer):
        stats = scorer.get_stats()
        assert "threshold" in stats
        assert "active_channels" in stats
        assert "total_active_signals" in stats
        assert stats["threshold"] == 0.7

    def test_velocity_history_zscore(self, scorer):
        channel = "testchannel"
        # Build up baseline history
        for _ in range(15):
            scorer.update_velocity_history(channel, 5.0)

        # A sudden spike should return high score
        spike_score = scorer.update_velocity_history(channel, 50.0)
        assert spike_score > 0.0

    def test_velocity_history_small_sample(self, scorer):
        # With < 10 samples, should use fixed threshold fallback
        score = scorer.update_velocity_history("newchannel", 25.0)
        assert score == pytest.approx(0.5)  # 25/50 = 0.5

    def test_virality_score_to_dict(self):
        signal = ViralitySignal(
            signal_type=SignalType.CHAT_VELOCITY,
            value=0.8,
            weight=0.3,
            timestamp=datetime.now(),
            source="kick",
            channel="test",
        )
        from detectors.virality_scorer import ViralityScore

        score = ViralityScore(
            score=0.75,
            is_viral=True,
            signals=[signal],
            timestamp=datetime.now(),
            channel="test",
            trigger_reason="chat_velocity: 0.80",
        )
        d = score.to_dict()
        assert d["score"] == 0.75
        assert d["is_viral"] is True
        assert d["signal_count"] == 1
        assert len(d["signals"]) == 1
