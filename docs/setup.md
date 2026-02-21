# Setup Guide

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime |
| FFmpeg | 6.0+ | Video processing |
| Chromium | Latest | Playwright automation |
| Git | 2.0+ | Version control |

## Installation

### 1. Clone and setup

```bash
git clone https://github.com/brolyroly007/Stake_downloader.git
cd Stake_downloader
python -m venv .venv
```

### 2. Activate virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Production only
make install

# Or with dev tools (recommended)
make dev
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values. At minimum, set:
- `KICK_CHANNELS` - Channels to monitor
- `GEMINI_API_KEY` - For AI features (get one at https://aistudio.google.com/apikey)

### 5. Install FFmpeg

**Windows:**
```bash
winget install Gyan.FFmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 6. Run

```bash
make run
# Open http://localhost:8000
```

## Docker Setup

```bash
# Copy and edit environment
cp .env.example .env

# Build and start
make docker-build
make docker-up

# View logs
docker compose logs -f

# Stop
make docker-down
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"
Make sure your virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### FFmpeg not found
Ensure FFmpeg is in your PATH:
```bash
ffmpeg -version
```

### Kick WebSocket connection fails
- Check your internet connection
- Try using a VPN if Kick is blocked in your region
- Check if `curl-cffi` is installed correctly

### Whisper model download fails
The first run downloads the Whisper model. Ensure you have internet access and sufficient disk space (~150MB for `base` model).

### Playwright browser not found
```bash
playwright install chromium
```

### Port 8000 already in use
```bash
python app.py --port 3000
```
