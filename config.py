"""
Configuración central del proyecto.
"""
from pathlib import Path
import os
import json
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Archivo para persistir canales
CHANNELS_FILE = Path(__file__).parent / "monitored_channels.json"

# ==========================================
# CONFIGURACIÓN DE CANALES
# ==========================================
def load_channels():
    """Carga canales desde archivo JSON."""
    if CHANNELS_FILE.exists():
        try:
            with open(CHANNELS_FILE, "r") as f:
                data = json.load(f)
                if data:
                    return data
        except Exception:
            pass
    # Defaults si no hay archivo
    return [
        "trainwreckstv",
        "roshtein",
        "xposed",
        "adinross",
        "buddha",
        "moose",
        "syztmz",
        "classybeef",
        "ayezee",
        "n3on",
    ]

def save_channels(channels):
    """Guarda canales en archivo JSON."""
    try:
        with open(CHANNELS_FILE, "w") as f:
            json.dump(channels, f, indent=2)
        return True
    except Exception:
        return False

# Lista de canales a monitorear
MONITORED_CHANNELS = load_channels()

# ==========================================
# UMBRALES DE VIRALIDAD
# ==========================================
# Velocidad de chat (mensajes por segundo) para considerar un spike
CHAT_VELOCITY_THRESHOLD = 50

# Score mínimo de viralidad (0.0 a 1.0) para descargar clip
VIRALITY_THRESHOLD = 0.7

# ==========================================
# CONFIGURACIÓN DE TIKTOK
# ==========================================
ENABLE_TIKTOK_UPLOAD = True
TIKTOK_HEADLESS_MODE = False  # False para ver el navegador (útil para debug/login)
TIKTOK_SESSION_FILE = Path("./tiktok_session.json")

# ==========================================
# GEMINI AI
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite"  # Modelo más ligero

# Habilitar análisis AI para filtrar clips
ENABLE_AI_FILTER = True
AI_MIN_VIRALITY_SCORE = 0.6  # Score mínimo para procesar clip

# ==========================================
# BUFFER DE STREAM
# ==========================================
ENABLE_STREAM_BUFFER = True
BUFFER_DURATION_SECONDS = 60  # Mantener 60s en buffer
PRE_SPIKE_SECONDS = 10  # Capturar 10s ANTES del spike
POST_SPIKE_SECONDS = 20  # Capturar 20s después del spike

# ==========================================
# RUTAS
# ==========================================
BASE_DIR = Path(__file__).parent
CLIPS_DIR = BASE_DIR / "clips"
RAW_CLIPS_DIR = CLIPS_DIR / "raw"
PROCESSED_CLIPS_DIR = CLIPS_DIR / "processed"

# Crear directorios si no existen
CLIPS_DIR.mkdir(exist_ok=True)
RAW_CLIPS_DIR.mkdir(exist_ok=True)
PROCESSED_CLIPS_DIR.mkdir(exist_ok=True)
