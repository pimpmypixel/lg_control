import asyncio
import os
import pickle
import time
from dotenv import load_dotenv
from playwright.async_api import Playwright, async_playwright, expect
from classes.detect.detect3 import LogoDetector

class Client:
    def __init__(self):
        load_dotenv()  # Load environment variables
        self.cookies_path = './storage/cookies.pkl'
        self.browser = None
        self.context = None
        self.page = None
        self.detector = None
        self.roi_image = None

    async def initialize(self, headless, roi_image):
        print(f"Init client - Headless: {headless} Image: {roi_image}")
        self.roi_image = roi_image
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            channel="chrome",
            headless=headless,
            args=['--app=https://play.tv2.dk/']
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
        await self._start_stream()

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
            
    async def _handle_login(self):
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
        time.sleep(1)
        await self.page.goto("https://play.tv2.dk/afspil/TV2NEWS")
        
        print("Start stream")
        # Mute
        try:
            await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').wait_for(timeout=3000)
            print("Mute stream")
            await self.page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').click()
        except Exception as e:
            print(f"Failed to find mute button: {e}")
            raise

        # Start logo detection
        self.detector = LogoDetector(roi_image=self.roi_image, roi_x=30, roi_y=10, roi_width=25, roi_height=25)
        await self.detector.start_detection(self.page)

    async def run(self):
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.cleanup()

    async def cleanup(self):
        if self.detector:
            await self.detector.stop_detection()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close() 