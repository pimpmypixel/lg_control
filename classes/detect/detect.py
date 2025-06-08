import asyncio
import cv2
import numpy as np
from PIL import Image
import io
import time

class LogoDetector:
    def __init__(self, roi_x=30, roi_y=30, roi_width=25, roi_height=25):
        self.roi_x = roi_x
        self.roi_y = roi_y  # Adjusted to account for window title bar
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.running = False
        self.detection_task = None
        print("Init detector")
        cv2.namedWindow('ROI Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ROI Detection', 400, 300)
        cv2.waitKey(1)  # Initialize window

    async def start_detection(self, page):
        """Start the detection process"""
        self.running = True
        print("Start detection")
        self.detection_task = asyncio.create_task(self._detect_loop(page))

    async def stop_detection(self):
        """Stop the detection process"""
        try:
            self.running = False
            if self.detection_task:
                self.detection_task.cancel()
                try:
                    await self.detection_task
                except (asyncio.CancelledError, Exception):
                    pass
            cv2.destroyAllWindows()
        except Exception:
            pass

    async def _detect_loop(self, page):
        """Main detection loop"""
        while self.running:
            try:
                # Take screenshot of the ROI
                screenshot = await page.screenshot()
                img = Image.open(io.BytesIO(screenshot))
                img_np = np.array(img)
                
                # Extract ROI
                roi = img_np[self.roi_y:self.roi_y+self.roi_height, 
                           self.roi_x:self.roi_x+self.roi_width]
                
                # Draw ROI rectangle on the full image
                cv2.rectangle(img_np, 
                            (self.roi_x, self.roi_y),
                            (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                            (0, 255, 0), 1)
                
                # Convert to grayscale for detection
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Apply adaptive thresholding to handle varying lighting conditions
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
                
                # Apply morphological operations to clean up noise
                kernel = np.ones((3,3), np.uint8)
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                # Apply Gaussian blur with smaller kernel for better edge preservation
                blurred = cv2.GaussianBlur(morph, (5, 5), 0)
                
                # Detect circles using Hough Circle Transform
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=5, maxRadius=10)

                if circles is None:
                    print("--ADS")
                else:
                    print("--TV")
                    # Draw detected circles
                    for i in circles[0, :]:
                        center = (int(i[0]), int(i[1]))
                        radius = int(i[2])
                        print(radius)
                        cv2.circle(roi, center, radius, (0, 255, 0), 1)
                        cv2.circle(roi, center, 2, (0, 0, 255), 1)

                # Show the image with ROI
                cv2.imshow('ROI Detection', img_np)
                cv2.waitKey(1)  # Update window

                await asyncio.sleep(0.5)  # Check more frequently

            except (asyncio.CancelledError, Exception):
                break
            except Exception as e:
                print(f"\nError in detection loop: {e}")
                await asyncio.sleep(1) 