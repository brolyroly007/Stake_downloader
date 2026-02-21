"""
Analizador de viralidad usando Gemini AI.

Usa Gemini Flash Lite (modelo más ligero) para:
- Analizar transcripciones de clips
- Detectar momentos virales
- Filtrar contenido no relevante
- Sugerir títulos para clips
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import aiohttp


class ViralityLevel(Enum):
    """Niveles de viralidad detectados por AI."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VIRAL = "viral"


@dataclass
class ClipAnalysis:
    """Resultado del análisis de un clip."""
    virality_score: float  # 0.0 a 1.0
    virality_level: ViralityLevel
    is_viral: bool
    reasons: List[str]
    suggested_title: str
    suggested_tags: List[str]
    content_type: str  # "gambling_win", "rage", "funny", "reaction", etc.
    confidence: float
    raw_response: Dict[str, Any]


class GeminiAnalyzer:
    """
    Analizador de clips usando Gemini Flash Lite.

    Analiza transcripciones y contexto para determinar
    si un clip tiene potencial viral.
    """

    # URL de la API de Gemini
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

    # Prompt del sistema para análisis de viralidad
    SYSTEM_PROMPT = """Eres un experto analizador de contenido viral para TikTok/Reels/Shorts.
Tu trabajo es analizar transcripciones de clips de streams de gambling/casino y determinar su potencial viral.

CRITERIOS DE VIRALIDAD ALTA:
- Grandes ganancias (big wins) con reacciones emocionales
- Momentos de tensión extrema (bonus buy, jackpot cerca)
- Reacciones explosivas del streamer (gritos, celebraciones)
- Fails épicos o pérdidas dramáticas
- Momentos graciosos o inesperados
- Interacciones memorables con el chat

CRITERIOS DE VIRALIDAD BAJA:
- Conversación normal sin emociones
- Gameplay rutinario sin eventos especiales
- Explicaciones técnicas largas
- Silencios o audio poco claro

Responde SIEMPRE en JSON con este formato exacto:
{
    "virality_score": 0.0-1.0,
    "is_viral": true/false,
    "reasons": ["razón 1", "razón 2"],
    "suggested_title": "título corto y llamativo",
    "suggested_tags": ["tag1", "tag2", "tag3"],
    "content_type": "tipo de contenido",
    "confidence": 0.0-1.0
}"""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: API key de Google Gemini
        """
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Obtiene o crea la sesión HTTP."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Cierra la sesión HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def analyze_transcript(
        self,
        transcript: str,
        channel: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ClipAnalysis]:
        """
        Analiza una transcripción para determinar viralidad.

        Args:
            transcript: Texto transcrito del clip
            channel: Nombre del canal/streamer
            context: Contexto adicional (chat velocity, viewers, etc.)

        Returns:
            ClipAnalysis o None si falla
        """
        # Construir el prompt de usuario
        user_prompt = self._build_user_prompt(transcript, channel, context)

        try:
            response = await self._call_gemini(user_prompt)
            if response:
                return self._parse_response(response)
            return None
        except Exception as e:
            print(f"[GeminiAnalyzer] Error: {e}")
            return None

    def _build_user_prompt(
        self,
        transcript: str,
        channel: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Construye el prompt para el usuario."""
        parts = []

        if channel:
            parts.append(f"CANAL/STREAMER: {channel}")

        if context:
            if "chat_velocity" in context:
                parts.append(f"VELOCIDAD DE CHAT: {context['chat_velocity']:.1f} msg/s")
            if "viewers" in context:
                parts.append(f"VIEWERS: {context['viewers']}")
            if "is_spike" in context and context["is_spike"]:
                parts.append("*** MOMENTO DE SPIKE DE CHAT DETECTADO ***")

        parts.append(f"\nTRANSCRIPCIÓN:\n{transcript}")
        parts.append("\nAnaliza el clip y responde en JSON.")

        return "\n".join(parts)

    async def _call_gemini(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Llama a la API de Gemini."""
        session = await self._get_session()

        url = f"{self.API_URL}?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.SYSTEM_PROMPT},
                        {"text": user_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500,
                "responseMimeType": "application/json"
            }
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extraer el texto de la respuesta
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            # Parsear JSON de la respuesta
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                # Intentar extraer JSON del texto
                                import re
                                json_match = re.search(r'\{[\s\S]*\}', text)
                                if json_match:
                                    return json.loads(json_match.group())
                    return None
                else:
                    error = await response.text()
                    print(f"[GeminiAnalyzer] API Error {response.status}: {error[:200]}")
                    return None

        except asyncio.TimeoutError:
            print("[GeminiAnalyzer] Timeout")
            return None
        except Exception as e:
            print(f"[GeminiAnalyzer] Request error: {e}")
            return None

    def _parse_response(self, response: Dict[str, Any]) -> ClipAnalysis:
        """Parsea la respuesta de Gemini a ClipAnalysis."""
        score = float(response.get("virality_score", 0.5))

        # Determinar nivel de viralidad
        if score >= 0.8:
            level = ViralityLevel.VIRAL
        elif score >= 0.6:
            level = ViralityLevel.HIGH
        elif score >= 0.4:
            level = ViralityLevel.MEDIUM
        else:
            level = ViralityLevel.LOW

        return ClipAnalysis(
            virality_score=score,
            virality_level=level,
            is_viral=response.get("is_viral", score >= 0.7),
            reasons=response.get("reasons", []),
            suggested_title=response.get("suggested_title", ""),
            suggested_tags=response.get("suggested_tags", []),
            content_type=response.get("content_type", "unknown"),
            confidence=float(response.get("confidence", 0.5)),
            raw_response=response,
        )

    async def analyze_with_audio(
        self,
        audio_path: Path,
        channel: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ClipAnalysis]:
        """
        Analiza un archivo de audio transcribiéndolo primero.

        Args:
            audio_path: Ruta al archivo de audio/video
            channel: Nombre del canal
            context: Contexto adicional

        Returns:
            ClipAnalysis o None
        """
        # Transcribir con Whisper
        transcript = await self._transcribe_audio(audio_path)
        if not transcript:
            return None

        return await self.analyze_transcript(transcript, channel, context)

    async def _transcribe_audio(self, audio_path: Path) -> Optional[str]:
        """Transcribe audio usando Whisper."""
        try:
            import whisper
        except ImportError:
            print("[GeminiAnalyzer] Whisper no instalado. pip install openai-whisper")
            return None

        try:
            # Cargar modelo pequeño para velocidad
            model = whisper.load_model("tiny")

            # Transcribir
            result = await asyncio.to_thread(
                model.transcribe,
                str(audio_path),
                language="es",
                fp16=False,
            )

            return result.get("text", "")

        except Exception as e:
            print(f"[GeminiAnalyzer] Transcription error: {e}")
            return None

    async def batch_analyze(
        self,
        clips: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List[Optional[ClipAnalysis]]:
        """
        Analiza múltiples clips en paralelo.

        Args:
            clips: Lista de dicts con {"transcript": str, "channel": str, "context": dict}
            max_concurrent: Máximo de análisis simultáneos

        Returns:
            Lista de ClipAnalysis (None para los que fallaron)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_one(clip: Dict) -> Optional[ClipAnalysis]:
            async with semaphore:
                return await self.analyze_transcript(
                    clip.get("transcript", ""),
                    clip.get("channel", ""),
                    clip.get("context"),
                )

        return await asyncio.gather(*[analyze_one(c) for c in clips])


class ViralityFilter:
    """
    Filtro de viralidad que combina análisis AI con métricas.

    Decide si un clip debe procesarse basándose en:
    - Análisis de Gemini AI
    - Velocidad de chat
    - Otros indicadores
    """

    def __init__(
        self,
        analyzer: GeminiAnalyzer,
        min_virality_score: float = 0.6,
        require_ai_approval: bool = True,
    ):
        self.analyzer = analyzer
        self.min_virality_score = min_virality_score
        self.require_ai_approval = require_ai_approval

    async def should_process_clip(
        self,
        transcript: str,
        channel: str,
        chat_velocity: float = 0.0,
        is_spike: bool = False,
        viewers: int = 0,
    ) -> tuple[bool, Optional[ClipAnalysis]]:
        """
        Decide si un clip debe ser procesado.

        Returns:
            Tuple de (should_process, analysis)
        """
        context = {
            "chat_velocity": chat_velocity,
            "is_spike": is_spike,
            "viewers": viewers,
        }

        # Si es un spike muy fuerte, aprobar sin AI
        if is_spike and chat_velocity > 100:
            # Spike muy alto, procesar de todas formas
            if not self.require_ai_approval:
                return True, None

        # Analizar con AI
        analysis = await self.analyzer.analyze_transcript(
            transcript,
            channel,
            context,
        )

        if analysis is None:
            # Si falla el AI, usar spike como fallback
            return is_spike, None

        # Decidir basándose en el score
        should_process = analysis.virality_score >= self.min_virality_score

        return should_process, analysis


# Función helper para uso rápido
async def quick_analyze(
    api_key: str,
    transcript: str,
    channel: str = "",
) -> Optional[ClipAnalysis]:
    """
    Análisis rápido de un transcript.

    Args:
        api_key: API key de Gemini
        transcript: Texto a analizar
        channel: Nombre del canal

    Returns:
        ClipAnalysis o None
    """
    analyzer = GeminiAnalyzer(api_key)
    try:
        return await analyzer.analyze_transcript(transcript, channel)
    finally:
        await analyzer.close()


# Ejemplo de uso
async def main():
    """Ejemplo de uso del analizador."""
    api_key = "AIzaSyC8Hydt3-fk0bimCyzZGpm3cZ03tJftAY4"

    analyzer = GeminiAnalyzer(api_key)

    # Ejemplo de transcripción
    test_transcript = """
    ¡VAMOS! ¡VAMOS! ¡Espera! ¿Qué es esto?
    ¡NO ME LO CREO! ¡500x! ¡QUINIENTAS EQUIS!
    ¡CHAT MIRA ESTO! ¡ES UN MEGA WIN!
    ¡Acabamos de ganar 50 mil dólares en un solo spin!
    ¡INCREÍBLE! ¡Esto es histórico!
    """

    try:
        print("Analizando transcripción...")
        analysis = await analyzer.analyze_transcript(
            test_transcript,
            channel="roshtein",
            context={"chat_velocity": 85.5, "is_spike": True, "viewers": 15000}
        )

        if analysis:
            print(f"\n=== RESULTADO ===")
            print(f"Score de viralidad: {analysis.virality_score:.2f}")
            print(f"Nivel: {analysis.virality_level.value}")
            print(f"¿Es viral?: {analysis.is_viral}")
            print(f"Razones: {', '.join(analysis.reasons)}")
            print(f"Título sugerido: {analysis.suggested_title}")
            print(f"Tags: {', '.join(analysis.suggested_tags)}")
            print(f"Tipo de contenido: {analysis.content_type}")
            print(f"Confianza: {analysis.confidence:.2f}")
        else:
            print("Error en el análisis")

    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())
