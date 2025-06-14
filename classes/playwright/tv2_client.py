import os
import time
from classes.playwright.base_client import BaseClient

class TV2Client(BaseClient):
    def __init__(self):
        super().__init__()
        self.stream_url = "https://play.tv2.dk/afspil/TV2NEWS"
        self.cookies_path = None
        self.detector = None
        self.app = None
        self.roi_image = None

    async def initialize(self, app: str, args):
        print(f"Initializing client for app: {app}")
        self.app = app
        self.cookies_path = f"./storage/{self.app}.pkl"
        await super().initialize(app, args)
            
    async def _handle_login(self):
        """Handle TV2-specific login process."""
        print("Starting login process...")
        await self.page.goto('https://play.tv2.dk')
        print("Navigated to TV2 website")

        time.sleep(1)
        try:
            await self.page.click("xpath=/html/body/div[1]/div/div[4]/div[1]/div/div[2]/button[1]")
            print("Closed cookies popup")
        except Exception as e:
            print(f"Could not close cookies popup: {e}")
        time.sleep(1)

        if await self.page.get_by_role("link", name="Log ind").is_visible():
            print("Need to log in")
            await self.page.get_by_role("link", name="Log ind").click()
            await self.page.fill("xpath=/html/body/div/div/div/div[3]/div[1]/form/div/div[1]/div/div/label/input", os.getenv('TV2_USERNAME'))
            await self.page.fill("xpath=/html/body/div/div/div/div[3]/div[1]/form/div/div[2]/div/label/input", os.getenv('TV2_PASSWORD'))
            time.sleep(1)
            print("Login credentials entered")

    async def _is_video_playing(self, video_selector):
        """Check if video is playing by examining the paused property"""
        is_paused =await self.page.evaluate(f"""
            () => {{
                const video = document.querySelector('{video_selector}');
                return video ? video.paused : null;
            }}
        """)
        return is_paused is False  

    async def _start_stream(self):
        print("Navigating to stream URL")
        await self.page.goto(self.stream_url)
        print("Waiting for stream to load...")
        time.sleep(2)
        
        # Mute
        try:
            print("Looking for mute button...")
            await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').wait_for(timeout=5000)
            print("Found mute button, clicking...")
            await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').click()
            print("Stream muted successfully")
        except Exception as e:
            print(f"Failed to find or click mute button: {e}")
            print('Waiting one minute before retrying...')
            time.sleep(60)
            raise

        if await self._is_video_playing("video"):
            print("Video is playing")
        else:
            print("Video is not playing")

        # Play
        try:
            await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/button[1]').wait_for(timeout=1000)
            await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/button[1]').click()
            print("Stream playing")
        except Exception as e:
            print(f"Failed to find or click play button: {e}")
            print('Waiting one minute before retrying...')
            time.sleep(60)
            raise

        return self.page