# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-02-21

### Added
- Real-time Kick.com stream monitoring via WebSocket (Pusher protocol)
- Real-time Stake.com monitoring via GraphQL polling
- Multi-signal virality detection system (chat velocity, audio peaks, big wins, emote spam)
- Google Gemini AI integration for content analysis and title suggestion
- Automatic clip downloading via yt-dlp
- Smart retroactive clipping (divide-and-conquer algorithm)
- Stream buffer for pre-spike capture (circular buffer)
- Video reframing to vertical (9:16) and square (1:1) formats
- Automatic caption generation via OpenAI Whisper
- TikTok upload via Playwright browser automation
- YouTube Shorts upload via Google API
- Instagram Reels upload via instagrapi
- FastAPI web dashboard with real-time WebSocket updates
- Channel management UI
- Clip library with player
- Manual capture and publish controls
- AI stream summaries per channel
- Pydantic-based configuration with .env support
- Structured logging with Loguru (file rotation)
- Cloudflare bypass with curl-cffi
- PyInstaller build support
