import pytest
from unittest.mock import MagicMock, patch
from publishers import TikTokUploader, YouTubeShortsUploader, InstagramReelsUploader

@pytest.mark.asyncio
async def test_tiktok_uploader_init():
    with patch('publishers.tiktok_uploader.async_playwright') as mock_playwright:
        # Mock the async context manager and start method
        mock_context = MagicMock()
        mock_playwright.return_value = mock_context
        
        # Make start() awaitable
        async def async_start():
            return MagicMock()
        mock_context.start.side_effect = async_start

        uploader = TikTokUploader()
        assert uploader is not None
        # Mock login check
        with patch.object(uploader, '_is_logged_in', return_value=True):
            # Mock browser init
            mock_page = MagicMock()
            # Make page methods awaitable
            async def async_goto(*args, **kwargs): return None
            async def async_wait_for_selector(*args, **kwargs): return None
            async def async_query_selector(*args, **kwargs): return None
            
            mock_page.goto.side_effect = async_goto
            mock_page.wait_for_selector.side_effect = async_wait_for_selector
            mock_page.query_selector.side_effect = async_query_selector
            
            uploader._browser = MagicMock()
            uploader._page = mock_page
            
            result = await uploader.login(wait_for_manual=False)
            assert result is True

@pytest.mark.asyncio
async def test_youtube_uploader_init():
    uploader = YouTubeShortsUploader()
    assert uploader is not None
    
    # Mock credentials
    with patch('publishers.youtube_shorts.Credentials') as mock_creds:
        with patch('publishers.youtube_shorts.build') as mock_build:
            with patch('pathlib.Path.exists', return_value=True):
                mock_creds.from_authorized_user_file.return_value = MagicMock(valid=True)
                result = uploader.authenticate()
                assert result is True

@pytest.mark.asyncio
async def test_instagram_uploader_init():
    with patch('publishers.instagram_reels.Client') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.login.return_value = True
        
        uploader = InstagramReelsUploader()
        # Mock config
        uploader.config.username = "test"
        uploader.config.password = "test"
        
        result = await uploader.login()
        assert result is True
