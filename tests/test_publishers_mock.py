import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
from publishers import TikTokUploader, YouTubeShortsUploader


@pytest.mark.asyncio
async def test_tiktok_uploader_init():
    with patch('publishers.tiktok_uploader.async_playwright') as mock_playwright:
        mock_context = MagicMock()
        mock_playwright.return_value = mock_context

        async def async_start():
            return MagicMock()
        mock_context.start.side_effect = async_start

        uploader = TikTokUploader()
        assert uploader is not None
        with patch.object(uploader, '_is_logged_in', return_value=True):
            mock_page = MagicMock()
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


def test_youtube_uploader_init():
    """Test YouTubeShortsUploader can be instantiated and has expected attributes."""
    uploader = YouTubeShortsUploader()
    assert uploader is not None
    assert uploader._credentials is None
    assert uploader._youtube is None
    assert uploader.config is not None
    assert uploader.config.client_secrets_file == Path("./client_secrets.json")


def test_youtube_uploader_auth_no_secrets():
    """authenticate returns False when client_secrets.json doesn't exist."""
    uploader = YouTubeShortsUploader()
    # client_secrets.json doesn't exist in test env, so auth should fail gracefully
    result = uploader.authenticate()
    assert result is False


@pytest.mark.asyncio
async def test_instagram_uploader_init():
    try:
        from publishers.instagram_reels import HAS_INSTAGRAPI
        if not HAS_INSTAGRAPI:
            pytest.skip("instagrapi not installed")
        from publishers import InstagramReelsUploader
    except ImportError:
        pytest.skip("instagrapi not installed")

    with patch('publishers.instagram_reels.Client') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.login.return_value = True

        uploader = InstagramReelsUploader()
        uploader.config.username = "test"
        uploader.config.password = "test"

        result = await uploader.login()
        assert result is True
