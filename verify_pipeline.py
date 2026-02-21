import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from pipeline import ViralPipeline, ViralMoment
from monitors.base_monitor import MonitorEvent, EventType
from publishers.tiktok_uploader import TikTokUploadResult

async def test_pipeline_integration():
    print("=== Testing Pipeline Integration ===")
    
    # 1. Setup Pipeline with mocks
    pipeline = ViralPipeline(
        output_dir=Path("./test_clips"),
        auto_download=True,
        auto_process=True,
        auto_upload=True
    )
    
    # Mock Downloader
    pipeline.downloader = MagicMock()
    pipeline.downloader.download = AsyncMock()
    # Mock return value for download (Clip object)
    mock_clip = MagicMock()
    mock_clip.file_path = Path("./test_clips/raw/test_video.mp4")
    pipeline.downloader.download.return_value = mock_clip
    
    # Mock Uploader
    pipeline.uploader = MagicMock()
    pipeline.uploader.upload = AsyncMock()
    pipeline.uploader.upload.return_value = TikTokUploadResult(
        success=True,
        video_url="https://tiktok.com/@user/video/123456"
    )
    pipeline.uploader._logged_in = True # Simulate logged in
    
    # Mock _download_live_stream to avoid subprocess
    pipeline._download_live_stream = AsyncMock()
    pipeline._download_live_stream.return_value = Path("./test_clips/raw/test_video.mp4")
    
    # Create dummy video file for validity
    mock_clip.file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_clip.file_path, "w") as f:
        f.write("dummy video content")

    # 2. Simulate Viral Event
    event = MonitorEvent(
        event_type=EventType.CHAT_SPIKE,
        source="kick",
        channel="test_channel",
        timestamp=datetime.now(),
        data={"velocity": 100}
    )
    
    print(f"1. Triggering event for channel: {event.channel}")
    moment = await pipeline.handle_event(event)
    
    # 3. Wait for processing
    print("2. Waiting for processing...")
    # Allow asyncio loop to process the queue
    await asyncio.sleep(1)
    
    # 4. Verify Results
    print(f"3. Moment status: {moment.status}")
    
    if moment.status == "published":
        print("✅ SUCCESS: Moment was published!")
        print(f"   URL: {moment.published_urls.get('tiktok')}")
    else:
        print(f"❌ FAILED: Moment status is {moment.status}")
        if moment.error:
            print(f"   Error: {moment.error}")

    # Verify calls
    # For Kick events, we expect _download_live_stream (or smart_clipper) to be called, not generic downloader
    if pipeline._download_live_stream.called:
        print("✅ _download_live_stream was called")
    else:
        print("⚠️ _download_live_stream was NOT called (maybe smart_clipper was used?)")
    
    pipeline.uploader.upload.assert_called_once()
    print("✅ Uploader was called")

    # Cleanup
    if mock_clip.file_path.exists():
        mock_clip.file_path.unlink()
    
if __name__ == "__main__":
    asyncio.run(test_pipeline_integration())
