import asyncio
import time
import pytest
from pathlib import Path
from recorders.clip_downloader import ClipDownloader, DownloadOptions

@pytest.mark.asyncio
async def test_concurrent_downloads():
    print("=== Testing Concurrent Downloads ===")
    
    output_dir = Path("./test_downloads")
    downloader = ClipDownloader(output_dir=output_dir)
    
    # URLs de prueba (usar clips pequeños o videos cortos)
    # Usaremos videos de prueba de YouTube que son cortos
    urls = [
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", # Me at the zoo (18s)
        "https://www.youtube.com/watch?v=BaW_jenozKc", # Test video 1
        "https://www.youtube.com/watch?v=aqz-KE-bpKQ", # Big Buck Bunny (short)
    ]
    
    start_time = time.time()
    
    print(f"Downloading {len(urls)} videos concurrently...")
    results = await downloader.download_multiple(urls, max_concurrent=3)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nTotal time: {duration:.2f}s")
    print(f"Successful downloads: {len(results)}")
    
    for clip in results:
        print(f"- {clip.title} ({clip.file_path.stat().st_size / 1024:.1f} KB)")
        # Clean up
        if clip.file_path.exists():
            clip.file_path.unlink()
        if clip.thumbnail_path and clip.thumbnail_path.exists():
            clip.thumbnail_path.unlink()
            
    # Clean up json files
    for f in output_dir.glob("*.json"):
        f.unlink()
        
    output_dir.rmdir()

if __name__ == "__main__":
    asyncio.run(test_concurrent_downloads())
