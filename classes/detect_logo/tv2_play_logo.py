import asyncio
import cv2
import numpy as np
from .base_detector import BaseDetector

class TV2PlayLogoDetector(BaseDetector):
    def __init__(self, roi_image, roi_x, roi_y, roi_width, roi_height):
        super().__init__(roi_image, roi_x, roi_y, roi_width, roi_height)
        # Red/blue color detection parameters (HSV ranges)
        # Adjusted for blue logo detection since the logo appears blue
        self.lower_blue = np.array([100, 150, 50])    # Blue range
        self.upper_blue = np.array([130, 255, 255])
        self.lower_red1 = np.array([0, 120, 70])      # Red range (backup)
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])    
        self.upper_red2 = np.array([180, 255, 255])
        # Detection thresholds
        self.min_red_pixels = 15  # Minimum red pixels to consider logo present
        self.min_contour_area = 20  # Minimum contour area for circular shape
        self.detection_threshold = 0.4  # Confidence threshold for detection
        print("Init detector")

    def detect_red_logo(self, roi):
        """
        Detect red/blue circular logo using color-based detection
        Returns detection confidence (0.0 to 1.0)
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create masks for both blue and red color ranges
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine blue and red masks
        combined_mask = cv2.bitwise_or(blue_mask, red_mask)
        
        # Count colored pixels
        colored_pixel_count = cv2.countNonZero(combined_mask)
        
        # Calculate base confidence from pixel count
        max_possible_pixels = roi.shape[0] * roi.shape[1]
        pixel_confidence = min(1.0, colored_pixel_count / (self.min_red_pixels * 2))
        
        # Enhanced confidence calculation for better balance
        if colored_pixel_count < self.min_red_pixels:
            # Calculate no-logo confidence based on pixel count and distribution
            # Lower pixel count = higher confidence of no logo
            no_logo_confidence = 1.0 - min(1.0, colored_pixel_count / self.min_red_pixels)
            
            # Add shape analysis for no-logo case
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shape_confidence = 0.0
            
            if contours:
                # Check if any contours are too small or irregular to be the logo
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_contour_area:
                        shape_confidence += 0.2  # Small contours increase no-logo confidence
                    else:
                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity < 0.5:  # Less circular = more likely not logo
                                shape_confidence += 0.3
            
            # Combine pixel and shape confidence for no-logo case
            final_confidence = (no_logo_confidence * 0.7) + (min(1.0, shape_confidence) * 0.3)
            return final_confidence, combined_mask, colored_pixel_count
        
        # Find contours in the mask to check for circular shapes
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return pixel_confidence * 0.5, combined_mask, colored_pixel_count
        
        # Check for circular contours and calculate shape confidence
        max_shape_confidence = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Calculate circularity (4*pi*area/perimeter^2)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Convert circularity to confidence (0.5 -> 0.5, 1.0 -> 1.0)
            shape_confidence = min(1.0, max(0.0, (circularity - 0.3) / 0.7))
            max_shape_confidence = max(max_shape_confidence, shape_confidence)
        
        # Combine pixel and shape confidence
        final_confidence = (pixel_confidence * 0.4) + (max_shape_confidence * 0.6)
        return final_confidence, combined_mask, colored_pixel_count

    async def render_roi_image(self):
        if self.roi_image:  # Using boolean flag correctly
            # Create a copy of the full screenshot for visualization
            display_img = self.full_screenshot.copy()
            
            # Draw ROI rectangle on display image
            color = (0, 255, 0) if self.logo_detected else (0, 0, 255)
            cv2.rectangle(display_img, 
                        (self.roi_x, self.roi_y),
                        (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                        color, 1)
            
            # Show the display image
            cv2.imshow('ROI Detection', display_img)
            cv2.waitKey(1)  # Add this to ensure the window updates

    async def _detect_loop(self, page):
        print("Starting detection loop...")
        
        while self.running:
            try:
                img_np = await self.make_screenshot(page)
                
                r, g, b = await self.average_color(img_np)
                roi = await self.make_roi(img_np)
                
                # Detect logo
                logo_confidence, color_mask, colored_pixels = self.detect_red_logo(roi)
                await self.publish_change(logo_confidence, colored_pixels)
                
                # Print debug info
                status = "TV" if self.logo_detected else "ADS"
                print(f"--{status} (conf: {logo_confidence:.2f}, running: {self.confidence:.2f}, "
                      f"pixels: {colored_pixels}, frames: {self.frames_since_state_change})", 
                      end='\r', flush=True)
                
                await self.render_image()
                await asyncio.sleep(self.cycle)

            except asyncio.CancelledError:
                print("\nDetection loop halt")
                break
            except Exception as e:
                print(f"\nError in detection loop: {e}")
                await asyncio.sleep(1)

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