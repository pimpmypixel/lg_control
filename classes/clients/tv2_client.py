import asyncio
import os
import pickle
import time
# from dotenv import load_dotenv
# from playwright.async_api import Playwright, async_playwright, expect
from classes.clients.base_client import BaseClient
from classes.detect.detect3 import LogoDetector

class Client(BaseClient):
    def __init__(self):
        super().__init__()
        self.stream_url = "https://play.tv2.dk/afspil/TV2NEWS"
        self.cookies_path = None
        self.detector = None
        self.app = None
        self.roi_image = None

    async def initialize(self, app: str, headless: bool, roi_image: bool):
        self.app = app
        self.cookies_path = f"./storage/{self.app}.pkl"
        self.roi_image = roi_image
        await super().initialize(app, headless, roi_image)
        await self._start_stream()
            
    async def _handle_login(self):
        """Handle TV2-specific login process."""
        print("Init login")
        await self.page.goto('https://play.tv2.dk')

        # Sales promo for unregistered users
        # if await self.page.is_visible("xpath=/html/body/div/div/div/div/div/div/div/div[1]/div[1]/button"):
        #     self.page.locator("xpath=/html/body/div/div/div/div/div/div/div/div[1]/div[1]/button").click()
        #     print("Closed popup")
        #     time.sleep(1)

        # Sales promo for unregistered users
        # if await self.page.is_visible("xpath=/html/body/div[1]/div/div[4]/div[1]/div/div[2]/button[1]"):
        time.sleep(1)
        await self.page.click("xpath=/html/body/div[1]/div/div[4]/div[1]/div/div[2]/button[1]")
        print("Closed cookies popup")
        time.sleep(1)

        # await self.page.locator("xpath=/html/body/div[3]/div[1]/div/div/div[2]/a").click()
        # print("Clicked login button")

        if await self.page.get_by_role("link", name="Log ind").is_visible():
            print("Need to log in")
            await self.page.get_by_role("link", name="Log ind").click()
            # await self.page.wait_for_selector("xpath=/html/body/div/div/div/div[3]/div[1]/form/div/div[1]/div/div/label/input", state="visible", timeout=5000)
            await self.page.fill("xpath=/html/body/div/div/div/div[3]/div[1]/form/div/div[1]/div/div/label/input", os.getenv('TV2_USERNAME'))
            await self.page.fill("xpath=/html/body/div/div/div/div[3]/div[1]/form/div/div[2]/div/label/input", os.getenv('TV2_PASSWORD'))
            # await self.page.get_by_role("button", name="Log ind").click()
            time.sleep(1)

            # Save cookies after successful login
        #     cookies = await self.context.cookies()
        #     with open(self.cookies_path, 'wb') as f:
        #         pickle.dump(cookies, f)
        #         print("Saved new cookies")
        # else:
        #     print("Already logged in")

    async def _start_stream(self):
        """Start the TV2 NEWS stream and initialize logo detection."""
        time.sleep(1)
        await self.page.goto(self.stream_url)
        
        print("Start stream")
        # Mute
        try:
            await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').wait_for(timeout=2000)
            print("Mute stream")
            await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').click()
        except Exception as e:
            print(f"Failed to find mute button: {e}")
            raise

        # Start logo detection
        self.detector = LogoDetector(roi_image=self.roi_image, roi_x=30, roi_y=10, roi_width=25, roi_height=25)
        await self.detector.start_detection(self.page)