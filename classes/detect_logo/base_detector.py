import asyncio
import os
import cv2
import numpy as np
import io
import pickle
from PIL import Image
from abc import ABC, abstractmethod
from playwright.async_api import Playwright, async_playwright
from dotenv import load_dotenv
from ..utils.message_bus import MessageBus, Message, MessageType

class BaseDetector(ABC):
    def __init__(self, roi_image: bool, roi_x, roi_y, roi_width, roi_height):
        self.cycle = 0.5
        self.roi_image = roi_image
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.running = False
        self.detection_task = None
        self.logo_detected = False 
        self.message_bus = MessageBus()
        self.last_detection_state = None  # Track previous state (True/False/None)
        self.frames_since_state_change = 0
        self.continuous_update_interval = 10  # Publish continuous updates every N frames
        self.consecutive_detections = 0
        self.consecutive_no_detections = 0
        self.render_frames_since_state_change = 0
        self.full_screenshot = None  # Store the full screenshot for visualization

        # Balanced confidence tracking
        self.confidence = 0.5  # Start at neutral confidence
        self.confidence_increment = 0.08  # Slightly reduced for smoother transitions
        self.confidence_decrement = 0.08  # Made equal to increment for balance
        self.max_confidence = 1.0
        self.min_confidence = 0.0
        
        # Initialize visualization window if ROI image is provided
        if self.roi_image is not False:
            cv2.namedWindow('ROI Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ROI Detection', 400, 300)
            cv2.waitKey(1)

    async def _publish_detection_update(self, logo_detected, logo_confidence, colored_pixels, is_state_change=False):
        """Publish detection updates with balanced information"""
        message_data = {
            'confidence': logo_confidence,
            'running_confidence': self.confidence,
            'pixels': colored_pixels,
            'detected': logo_detected,
            'state_change': is_state_change,
            'frames_since_change': self.frames_since_state_change
        }
        
        if logo_detected:
            message_type = MessageType.LOGO_DETECTED
        else:
            message_type = MessageType.LOGO_LOST
            
        await self.message_bus.publish(Message(message_type, message_data))
    
    async def make_screenshot(self, page):
        screenshot = await page.screenshot()
        img = Image.open(io.BytesIO(screenshot))
        img_np = np.array(img)
        # Convert RGB to BGR for OpenCV
        self.full_screenshot = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return self.full_screenshot
    
    async def average_color(self, img_np):
        avg_bgr = np.mean(img_np, axis=(0, 1))  # Average across both height and width
        avg_rgb = avg_bgr[::-1]  # Convert BGR to RGB
        return int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])
    

    async def make_roi(self, img_np):
        # Extract ROI directly from the full image
        return img_np[self.roi_y:self.roi_y+self.roi_height, 
                     self.roi_x:self.roi_x+self.roi_width]

    async def start_detection(self, page):
        """Start the detection process"""
        self.running = True
        print("Init detection")
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

    async def publish_change(self, logo_confidence, colored_pixels): 
        # Update logo detection state based on confidence threshold
        self.logo_detected = logo_confidence > self.detection_threshold
                
        # Update running confidence with balanced smoothing
        # Use the actual confidence value to influence the running confidence
        if self.logo_detected:
            # When logo is detected, increase confidence more if the detection confidence is high
            confidence_factor = min(1.0, logo_confidence * 1.5)  # Amplify high confidence detections
            self.confidence = min(self.max_confidence, self.confidence + (self.confidence_increment * confidence_factor))
        else:
            # When no logo is detected, decrease confidence more if the no-logo confidence is high
            confidence_factor = min(1.0, logo_confidence * 1.5)  # Amplify high confidence no-detections
            self.confidence = max(self.min_confidence, self.confidence - (self.confidence_decrement * confidence_factor))
        
        # Track state changes and frame counts
        state_changed = False
        if self.last_detection_state is None:
            # First detection
            self.last_detection_state = self.logo_detected
            state_changed = True
            self.frames_since_state_change = 0
        elif self.last_detection_state != self.logo_detected:
            # State changed
            state_changed = True
            self.last_detection_state = self.logo_detected
            self.frames_since_state_change = 0
        else:
            self.frames_since_state_change += 1
            self.render_frames_since_state_change += 1
        
        # Use consecutive detection logic for stability
        should_publish = False
        if self.logo_detected:
            self.consecutive_detections += 1
            self.consecutive_no_detections = 0
            if self.consecutive_detections >= 2:  # Require 2 consecutive detections
                should_publish = True
        else:
            self.consecutive_no_detections += 1
            self.consecutive_detections = 0
            if self.consecutive_no_detections >= 2:  # Require 2 consecutive non-detections
                should_publish = True
        
        # Publish updates: on state changes OR periodic continuous updates
        if should_publish and (state_changed or 
                                self.frames_since_state_change % self.continuous_update_interval == 0):
            await self._publish_detection_update(
                self.logo_detected, logo_confidence, colored_pixels, state_changed
            )

    async def render_image(self):
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