# Architecture

## System Overview

Stake Downloader follows an **event-driven pipeline architecture** where each component operates independently and communicates through events and callbacks.

## Core Components

### 1. Monitors (`monitors/`)

Monitors are responsible for connecting to streaming platforms and detecting events in real-time.

```
BaseMonitor (abstract)
├── KickMonitor    - WebSocket via Pusher protocol
└── StakeMonitor   - GraphQL polling + WebSocket
```

**Key concepts:**
- Each monitor runs an async listen loop
- Events are emitted via `_emit_event()` to registered handlers
- Automatic reconnection on connection loss
- Chat velocity tracking with configurable thresholds

### 2. Detectors (`detectors/`)

Detectors analyze events from monitors and score them for virality.

**ViralityScorer** uses weighted multi-signal scoring:
- Maintains a sliding window of signals (default 30s)
- Calculates Z-scores for adaptive threshold detection
- Enforces cooldown between triggers to avoid duplicates

**GeminiAnalyzer** provides AI-powered content analysis:
- Transcription analysis via Google Gemini API
- Title and tag suggestion
- Virality confidence scoring

### 3. Pipeline (`pipeline.py`)

The pipeline orchestrates the full processing flow:

```
Event Received
    ↓
Download clip (yt-dlp / SmartClipper)
    ↓
AI Analysis (Gemini) → Filter low-score clips
    ↓
Reframe to vertical/square (OpenCV + MoviePy)
    ↓
Generate captions (Whisper)
    ↓
Mark as ready → Notify callbacks
```

**Concurrency:** Processes up to 3 clips simultaneously using `asyncio.Semaphore`.

### 4. Recorders (`recorders/`)

Multiple strategies for capturing video:

| Strategy | Use Case | Approach |
|----------|----------|----------|
| `ClipDownloader` | Standard clips | yt-dlp download |
| `SmartClipper` | Live stream spikes | Divide-and-conquer algorithm |
| `StreamBuffer` | Pre-spike capture | Circular buffer (60s rolling) |
| `ReplayRenderer` | Stake replays | Playwright screenshot capture |

### 5. Processors (`processors/`)

Video processing modules:

- **VideoReframer**: Converts 16:9 to 9:16 or 1:1 using face tracking or center crop
- **CaptionGenerator**: Transcribes audio with Whisper and burns subtitles into video
- **VideoCompiler**: Assembles final video with all processing applied

### 6. Publishers (`publishers/`)

Upload processed clips to social media:

| Platform | Method | Auth |
|----------|--------|------|
| TikTok | Playwright browser automation | Session cookies |
| YouTube Shorts | Google API (v3) | OAuth2 credentials |
| Instagram Reels | instagrapi library | Username/password |

### 7. Web Dashboard (`app.py` + `web/`)

FastAPI application with WebSocket real-time updates:

- **REST API**: CRUD operations for channels, clips, config
- **WebSocket**: Live stats, event feed, log streaming
- **Frontend**: Single-page app with vanilla JS + Tailwind CSS

## Data Flow

```
┌─────────┐    Events    ┌──────────┐   Score   ┌──────────┐
│ Monitor │───────────→ │ Detector │────────→ │ Pipeline │
└─────────┘              └──────────┘           └────┬─────┘
                                                     │
                              ┌───────────────────────┤
                              ↓           ↓           ↓
                         ┌────────┐  ┌────────┐  ┌────────┐
                         │Download│  │Process │  │Publish │
                         └────────┘  └────────┘  └────────┘
```

## Configuration

Configuration flows through two systems:

1. **`config.py`** (root) - Simple module-level config with `os.getenv()` fallbacks
2. **`core/config.py`** - Pydantic `BaseSettings` with validation and `.env` file support

The root `config.py` is used by `app.py` and `pipeline.py` for quick access.
The `core/config.py` provides type-safe configuration for internal modules.

## Error Handling

- Monitors auto-reconnect on connection loss (5s backoff)
- Pipeline catches and logs errors per-moment without crashing
- Failed clips are marked with `status="failed"` and `error` message
- AI-filtered clips are cleaned up from disk to save space
