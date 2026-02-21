import asyncio
import shutil
import time
import pytest
from pathlib import Path
from recorders.clip_downloader import ClipDownloader, DownloadOptions


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_downloads(tmp_path):
    """Test concurrent downloads with yt-dlp (requires network)."""
    downloader = ClipDownloader(output_dir=tmp_path)

    urls = [
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Me at the zoo (18s)
    ]

    start_time = time.time()
    results = await downloader.download_multiple(urls, max_concurrent=3)
    duration = time.time() - start_time

    assert len(results) >= 1
    for clip in results:
        assert clip.file_path.exists()
        assert clip.file_path.stat().st_size > 0


if __name__ == "__main__":
    asyncio.run(test_concurrent_downloads())
