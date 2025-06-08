import os
import time
import ffmpeg
import asyncio
import subprocess
from classes.clients.base_client import BaseClient
from classes.detect.detect3 import LogoDetector

class Client(BaseClient):
    def __init__(self):
        super().__init__()
        # self.stream_url = "https://play.tv2.dk/afspil/TV2NEWS"
        self.stream_url = "https://play.tv2.dk/afspil/TV2ECHO"
        self.cookies_path = None
        self.detector = None
        self.app = None
        self.roi_image = None

    async def initialize(self, app: str, args):
        self.app = app
        self.cookies_path = f"./storage/{self.app}.pkl"
        self.roi_image = args.roi
        self.audio = args.audio
        await super().initialize(app, args)
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

    async def _start_stream(self):
        """Start the TV2 NEWS stream and initialize logo detection."""
        time.sleep(1)
        await self.page.goto(self.stream_url)
        
        print("Start stream")
        # Mute
        try:
            if self.audio is False:
                await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').wait_for(timeout=2000)
                print("Mute stream")
                await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').click()
        except Exception as e:
            print(f"Failed to find mute button: {e}")
            print('Wait one minute')
            raise


        # duration=0
        # output_file='./recordings/test.mp3'
        # cmd = [
        #     'ffmpeg',
        #     '-f', 'avfoundation',
        #     '-i', ':0',  # Default audio input
        #     '-t', str(duration),
        #     '-acodec', 'mp3',
        #     output_file
        # ]

        # print(f"Recording audio for {duration} seconds...")
        # try:
        #     process = await asyncio.create_subprocess_exec(
        #         *cmd,
        #         stdout=asyncio.subprocess.PIPE,
        #         stderr=asyncio.subprocess.PIPE
        #     )
        #     stdout, stderr = await process.communicate()
            
        #     if process.returncode == 0:
        #         print(f"Audio saved to {output_file}")
        #     else:
        #         print(f"FFmpeg error: {stderr.decode()}")
                
        # except FileNotFoundError:
        #     print("FFmpeg not found. Install with: brew install ffmpeg")
        # except Exception as e:
        #     print(f"An error occurred: {e}")

        # Start logo detection
        self.detector = LogoDetector(roi_image=self.roi_image, roi_x=30, roi_y=10, roi_width=25, roi_height=25)
        await self.detector.start_detection(self.page)