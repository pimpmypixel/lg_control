import asyncio
import os
import pickle
from abc import ABC, abstractmethod
from playwright.async_api import Playwright, async_playwright
from dotenv import load_dotenv

class BaseClient(ABC):
    def __init__(self):
        load_dotenv()
        self.cookies_path = None
        self.app = None
        self.browser = None
        self.context = None
        self.page = None

    @abstractmethod
    async def initialize(self, app, headless, roi_image):
        """Initialize client with browser setup and logo detection.
        
        Args:
            app (str): LG tv app name
            headless (bool): Whether to run the browser in headless mode
            roi_image (str): Path to the ROI image for logo detection
        """
        print(f"Init client - Headless: {headless} - ROI image: {roi_image}")
        playwright = await async_playwright().start()

        self.browser = await playwright.chromium.launch(
            channel="chrome",
            headless=headless,
        )

        self.context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            viewport={'width': 600, 'height': 338},  # 16:9 aspect ratio for 600px width
            permissions=['notifications'],
            ignore_https_errors=True,
            java_script_enabled=True,
            has_touch=False,
            is_mobile=False,
            locale='da-DK',
            timezone_id='Europe/Copenhagen',
            geolocation={'latitude': 55.676098, 'longitude': 12.568337},
            color_scheme='dark',
            reduced_motion='no-preference',
            forced_colors='none',
            accept_downloads=False,
            extra_http_headers={
                'Accept-Language': 'da-DK,en;q=0.9',
            }
        )

        self.page = await self.context.new_page()
        if not await self._load_cookies():
            await self._handle_login()

    async def _load_cookies(self):
        if os.path.exists(self.cookies_path):
            with open(self.cookies_path, 'rb') as f:
                cookies = pickle.load(f)
                print("Adding found cookies")
                await self.context.add_cookies(cookies)
                return True
        else:
            print("No cookies found")
            return False

    @abstractmethod
    async def _handle_login(self):
        """Handle the login process for the specific service.
        This method must be implemented by subclasses.
        """
        pass

    async def run(self):
        """Main run loop for the client."""
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Exit")
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources when the client is done.
        This method must be implemented by subclasses.
        """
        if self.detector:
            await self.detector.stop_detection()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close() 
        print("Done cleanup")