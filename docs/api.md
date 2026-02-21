# API Reference

Base URL: `http://localhost:8000`

## REST Endpoints

### System

#### `GET /api/status`
Returns the current system status.

**Response:**
```json
{
  "is_running": true,
  "channels": [
    {
      "slug": "trainwreckstv",
      "chatroom_id": 12345,
      "is_live": true,
      "viewer_count": 45000,
      "velocity": 12.5
    }
  ],
  "stats": {
    "messages": 15420,
    "spikes": 3,
    "clips_ready": 2,
    "start_time": "2025-01-15T14:30:00"
  },
  "events": []
}
```

#### `GET /api/config`
Returns the current configuration.

#### `POST /api/start`
Starts monitoring all configured channels.

#### `POST /api/stop`
Stops all monitoring.

---

### Channels

#### `POST /api/channels`
Updates the list of monitored channels.

**Body:** `["channel1", "channel2", "channel3"]`

**Response:**
```json
{
  "status": "updated",
  "channels": ["channel1", "channel2", "channel3"]
}
```

---

### Clips

#### `GET /api/clips`
Lists all captured clips with pipeline stats.

#### `GET /api/clips/{clip_id}`
Returns details for a specific clip.

#### `POST /api/capture/{channel}?duration=30`
Manually captures a clip from a live channel.

**Parameters:**
- `channel` (path): Channel slug
- `duration` (query, optional): Capture duration in seconds (default: 30)

#### `POST /api/publish/{clip_id}/{platform}`
Publishes a clip to a social media platform.

**Parameters:**
- `clip_id` (path): The clip identifier
- `platform` (path): `tiktok`, `youtube`, or `instagram`

---

### Pipeline

#### `GET /api/pipeline/status`
Returns the pipeline processing status and recent moments.

---

### Stream Buffer

#### `POST /api/buffer/start/{channel}`
Starts the stream buffer for continuous capture.

#### `POST /api/buffer/stop/{channel}`
Stops the stream buffer.

#### `GET /api/buffer/status`
Returns status of all active buffers.

#### `POST /api/buffer/capture/{channel}?pre_seconds=10&post_seconds=20`
Captures a clip from the buffer including pre-event context.

---

### AI Analysis

#### `GET /api/ai/summaries`
Returns AI-generated summaries for all monitored streams.

#### `GET /api/ai/summary/{channel}`
Returns detailed AI summary for a specific channel.

#### `POST /api/ai/analyze`
Analyzes a transcript with Gemini AI.

**Body:**
```json
{
  "transcript": "text to analyze",
  "channel": "channel_name"
}
```

---

### Streamers

#### `GET /api/my-streamers`
Returns the status of favorite streamers (live/offline).

#### `GET /api/live-streamers`
Returns all currently live streamers from a curated list.

---

## WebSocket

### `WS /ws`

Connect for real-time updates.

**Message types received:**

| Type | Description |
|------|-------------|
| `init` | Initial state on connection |
| `stats` | Periodic stats update |
| `event` | Chat spike detected |
| `clip_ready` | Clip finished processing |
| `clip_analyzed` | AI analysis completed |
| `stream_summary` | Updated stream summary |
| `log` | Log message |
| `status` | System status change |

**Example message:**
```json
{
  "type": "event",
  "data": {
    "type": "spike",
    "channel": "trainwreckstv",
    "velocity": 85.3,
    "timestamp": "2025-01-15T14:35:22"
  }
}
```
