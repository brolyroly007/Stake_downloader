"""
Sistema de puntuación de viralidad.

Combina múltiples señales para determinar si un momento
es potencialmente viral y debería ser capturado.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio


class SignalType(Enum):
    """Tipos de señales de viralidad."""
    CHAT_VELOCITY = "chat_velocity"      # Pico de mensajes en chat
    AUDIO_PEAK = "audio_peak"            # Gritos/reacciones de audio
    BIG_WIN = "big_win"                  # Victoria grande en casino
    EMOTE_SPAM = "emote_spam"            # Spam de emotes específicos
    VIEWER_SPIKE = "viewer_spike"        # Aumento repentino de viewers
    DONATION = "donation"                # Donación grande
    RAID = "raid"                        # Raid de otro canal
    CLIP_REQUESTED = "clip_requested"    # Alguien pidió clip (!clip)
    CUSTOM = "custom"                    # Señal personalizada


@dataclass
class ViralitySignal:
    """
    Representa una señal individual de viralidad.
    """
    signal_type: SignalType
    value: float          # Valor normalizado (0-1)
    weight: float         # Peso en el score final (0-1)
    timestamp: datetime
    source: str           # kick, stake, audio, etc.
    channel: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_value: Optional[float] = None  # Valor original sin normalizar

    @property
    def weighted_value(self) -> float:
        """Valor ponderado por el peso."""
        return self.value * self.weight


@dataclass
class ViralityScore:
    """
    Resultado del cálculo de viralidad.
    """
    score: float                    # Score total (0-1)
    is_viral: bool                  # Si supera el umbral
    signals: List[ViralitySignal]   # Señales que contribuyeron
    timestamp: datetime
    channel: str
    trigger_reason: str             # Razón principal del trigger

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para logging/storage."""
        return {
            "score": self.score,
            "is_viral": self.is_viral,
            "channel": self.channel,
            "trigger_reason": self.trigger_reason,
            "timestamp": self.timestamp.isoformat(),
            "signal_count": len(self.signals),
            "signals": [
                {
                    "type": s.signal_type.value,
                    "value": s.value,
                    "weight": s.weight,
                    "source": s.source,
                }
                for s in self.signals
            ]
        }


class ViralityScorer:
    """
    Calcula puntuaciones de viralidad basadas en múltiples señales.

    El sistema usa una combinación ponderada de señales para
    determinar si un momento es "viral" y debería ser capturado.

    Configuración de pesos por defecto:
    - Chat Velocity: 0.30 (30%)
    - Audio Peak: 0.25 (25%)
    - Big Win: 0.25 (25%)
    - Emote Spam: 0.20 (20%)
    """

    DEFAULT_WEIGHTS = {
        SignalType.CHAT_VELOCITY: 0.30,
        SignalType.AUDIO_PEAK: 0.25,
        SignalType.BIG_WIN: 0.25,
        SignalType.EMOTE_SPAM: 0.20,
        SignalType.VIEWER_SPIKE: 0.15,
        SignalType.DONATION: 0.10,
        SignalType.RAID: 0.15,
        SignalType.CLIP_REQUESTED: 0.20,
        SignalType.CUSTOM: 0.10,
    }

    # Umbrales para normalización de valores
    NORMALIZATION_THRESHOLDS = {
        SignalType.AUDIO_PEAK: {
            "min": -60,    # dB (silencio)
            "max": 0,      # dB (muy alto)
        },
        SignalType.BIG_WIN: {
            "min": 0,      # USD
            "max": 100000, # USD
        },
    }

    def __init__(
        self,
        threshold: float = 0.7,
        weights: Optional[Dict[SignalType, float]] = None,
        signal_window_seconds: float = 30.0,
        cooldown_seconds: float = 60.0,
    ):
        """
        Args:
            threshold: Score mínimo para considerar viral (0-1)
            weights: Pesos personalizados por tipo de señal
            signal_window_seconds: Ventana de tiempo para agregar señales
            cooldown_seconds: Cooldown entre triggers virales
        """
        self.threshold = threshold
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.signal_window = signal_window_seconds
        self.cooldown = cooldown_seconds

        # Señales activas por canal
        self._active_signals: Dict[str, List[ViralitySignal]] = {}

        # Último trigger por canal (para cooldown)
        self._last_trigger: Dict[str, datetime] = {}

        # Callbacks para cuando se detecta viralidad
        self._viral_callbacks: List[callable] = []
        
        # Historial de velocidad para Z-score (adaptive threshold)
        # channel -> list of velocity values
        self._velocity_history: Dict[str, List[float]] = {}
        self._history_max_size = 100  # Mantener últimos 100 puntos

    def register_callback(self, callback: callable) -> None:
        """Registra un callback para cuando se detecta viralidad."""
        self._viral_callbacks.append(callback)

    def add_signal(self, signal: ViralitySignal) -> Optional[ViralityScore]:
        """
        Añade una señal y calcula el score de viralidad.

        Args:
            signal: Señal de viralidad detectada

        Returns:
            ViralityScore si se supera el umbral, None si no
        """
        channel = signal.channel

        # Inicializar lista de señales para el canal
        if channel not in self._active_signals:
            self._active_signals[channel] = []

        # Limpiar señales fuera de la ventana de tiempo
        self._cleanup_old_signals(channel)

        # Añadir nueva señal
        self._active_signals[channel].append(signal)

        # Calcular score
        score = self._calculate_score(channel)

        # Verificar si es viral
        if score.is_viral:
            # Verificar cooldown
            if self._is_in_cooldown(channel):
                score.is_viral = False
                score.trigger_reason = f"En cooldown (score: {score.score:.2f})"
            else:
                # Marcar trigger
                self._last_trigger[channel] = datetime.now()

                # Llamar callbacks
                for callback in self._viral_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(score))
                        else:
                            callback(score)
                    except Exception:
                        pass

        return score if score.is_viral else None

    def _cleanup_old_signals(self, channel: str) -> None:
        """Elimina señales fuera de la ventana de tiempo."""
        if channel not in self._active_signals:
            return

        now = datetime.now()
        cutoff = now.timestamp() - self.signal_window

        self._active_signals[channel] = [
            s for s in self._active_signals[channel]
            if s.timestamp.timestamp() > cutoff
        ]

    def _calculate_score(self, channel: str) -> ViralityScore:
        """Calcula el score de viralidad para un canal."""
        signals = self._active_signals.get(channel, [])

        if not signals:
            return ViralityScore(
                score=0.0,
                is_viral=False,
                signals=[],
                timestamp=datetime.now(),
                channel=channel,
                trigger_reason="No signals",
            )

        # Agrupar señales por tipo y tomar el máximo de cada tipo
        max_by_type: Dict[SignalType, ViralitySignal] = {}

        for signal in signals:
            sig_type = signal.signal_type
            if sig_type not in max_by_type:
                max_by_type[sig_type] = signal
            elif signal.weighted_value > max_by_type[sig_type].weighted_value:
                max_by_type[sig_type] = signal

        # Calcular score total
        total_score = 0.0
        total_weight = 0.0
        contributing_signals = []

        for sig_type, signal in max_by_type.items():
            weight = self.weights.get(sig_type, 0.1)
            total_score += signal.value * weight
            total_weight += weight
            contributing_signals.append(signal)

        # Normalizar por peso total
        if total_weight > 0:
            # No normalizar completamente, permitir scores altos con muchas señales
            final_score = min(total_score, 1.0)
        else:
            final_score = 0.0

        # Determinar razón principal
        if contributing_signals:
            top_signal = max(contributing_signals, key=lambda s: s.weighted_value)
            trigger_reason = f"{top_signal.signal_type.value}: {top_signal.value:.2f}"
        else:
            trigger_reason = "Unknown"

        return ViralityScore(
            score=final_score,
            is_viral=final_score >= self.threshold,
            signals=contributing_signals,
            timestamp=datetime.now(),
            channel=channel,
            trigger_reason=trigger_reason,
        )

    def _is_in_cooldown(self, channel: str) -> bool:
        """Verifica si el canal está en cooldown."""
        if channel not in self._last_trigger:
            return False

        elapsed = (datetime.now() - self._last_trigger[channel]).total_seconds()
        return elapsed < self.cooldown

    def normalize_value(
        self,
        signal_type: SignalType,
        raw_value: float
    ) -> float:
        """
        Normaliza un valor crudo a rango 0-1.

        Args:
            signal_type: Tipo de señal
            raw_value: Valor original

        Returns:
            Valor normalizado entre 0 y 1
        """
        thresholds = self.NORMALIZATION_THRESHOLDS.get(signal_type)

        if not thresholds:
            # Sin thresholds definidos, asumir 0-100
            return min(max(raw_value / 100.0, 0.0), 1.0)

        min_val = thresholds["min"]
        max_val = thresholds["max"]

        # Normalizar
        if max_val == min_val:
            return 0.5

        normalized = (raw_value - min_val) / (max_val - min_val)
        return min(max(normalized, 0.0), 1.0)

    def update_velocity_history(self, channel: str, velocity: float) -> float:
        """
        Actualiza el historial y retorna un score basado en Z-score.
        Esto permite detectar picos relativos al promedio del canal.
        """
        if channel not in self._velocity_history:
            self._velocity_history[channel] = []
        
        history = self._velocity_history[channel]
        history.append(velocity)
        if len(history) > self._history_max_size:
            history.pop(0)
            
        # Necesitamos un mínimo de datos para calcular estadísticas confiables
        if len(history) < 10:
            # Fallback a threshold fijo simple mientras aprendemos
            return min(velocity / 50.0, 1.0)
            
        # Calcular media y desviación estándar
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
            
        # Z-score: cuántas desviaciones estándar por encima de la media
        z_score = (velocity - mean) / std_dev
        
        # Normalizar Z-score a 0-1
        # Asumimos que un Z-score de 3 (3 sigma) es un pico muy alto (1.0)
        # Z-score < 1 es ruido normal (0.0)
        if z_score < 1.0:
            return 0.0
        
        normalized_score = (z_score - 1.0) / 3.0
        return min(max(normalized_score, 0.0), 1.0)

    def create_signal(
        self,
        signal_type: SignalType,
        raw_value: float,
        source: str,
        channel: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ViralitySignal:
        """
        Crea una señal normalizada.

        Args:
            signal_type: Tipo de señal
            raw_value: Valor sin normalizar
            source: Fuente de la señal
            channel: Canal/stream
            metadata: Datos adicionales

        Returns:
            ViralitySignal lista para usar
        """
        if signal_type == SignalType.CHAT_VELOCITY:
            # Usar lógica adaptativa para chat velocity
            normalized = self.update_velocity_history(channel, raw_value)
        else:
            normalized = self.normalize_value(signal_type, raw_value)
            
        weight = self.weights.get(signal_type, 0.1)

        return ViralitySignal(
            signal_type=signal_type,
            value=normalized,
            weight=weight,
            timestamp=datetime.now(),
            source=source,
            channel=channel,
            metadata=metadata or {},
            raw_value=raw_value,
        )

    def get_current_score(self, channel: str) -> ViralityScore:
        """Obtiene el score actual de un canal sin añadir señales."""
        self._cleanup_old_signals(channel)
        return self._calculate_score(channel)

    def get_active_channels(self) -> List[str]:
        """Retorna canales con señales activas."""
        return list(self._active_signals.keys())

    def clear_channel(self, channel: str) -> None:
        """Limpia todas las señales de un canal."""
        if channel in self._active_signals:
            del self._active_signals[channel]

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del scorer."""
        return {
            "threshold": self.threshold,
            "signal_window_seconds": self.signal_window,
            "cooldown_seconds": self.cooldown,
            "active_channels": len(self._active_signals),
            "total_active_signals": sum(
                len(signals) for signals in self._active_signals.values()
            ),
            "channels_in_cooldown": sum(
                1 for ch in self._active_signals
                if self._is_in_cooldown(ch)
            ),
        }


# =====================================================
# EJEMPLO DE USO
# =====================================================

async def main():
    """Ejemplo de uso del ViralityScorer."""
    from core.logger import logger

    scorer = ViralityScorer(
        threshold=0.6,
        cooldown_seconds=30.0,
    )

    # Callback cuando se detecta viralidad
    async def on_viral(score: ViralityScore):
        logger.info(
            f"🔥 VIRAL DETECTED!\n"
            f"   Channel: {score.channel}\n"
            f"   Score: {score.score:.2f}\n"
            f"   Reason: {score.trigger_reason}\n"
            f"   Signals: {len(score.signals)}"
        )

    scorer.register_callback(on_viral)

    # Simular señales
    channel = "trainwreckstv"

    # Señal de chat velocity
    signal1 = scorer.create_signal(
        signal_type=SignalType.CHAT_VELOCITY,
        raw_value=75,  # 75 msgs/sec
        source="kick",
        channel=channel,
    )
    result = scorer.add_signal(signal1)
    logger.info(f"Chat velocity signal: score = {scorer.get_current_score(channel).score:.2f}")

    # Señal de Big Win
    signal2 = scorer.create_signal(
        signal_type=SignalType.BIG_WIN,
        raw_value=50000,  # $50k win
        source="stake",
        channel=channel,
    )
    result = scorer.add_signal(signal2)
    logger.info(f"Big win signal: score = {scorer.get_current_score(channel).score:.2f}")

    # Señal de audio
    signal3 = scorer.create_signal(
        signal_type=SignalType.AUDIO_PEAK,
        raw_value=-10,  # -10 dB (loud)
        source="audio",
        channel=channel,
    )
    result = scorer.add_signal(signal3)

    if result:
        logger.info(f"Final viral score: {result.score:.2f}")

    # Estadísticas
    logger.info(f"Stats: {scorer.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
