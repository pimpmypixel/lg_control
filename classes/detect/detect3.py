import asyncio
import cv2
import numpy as np
from PIL import Image
import io
from ..utils.message_bus import MessageBus, Message, MessageType

class LogoDetector:
    def __init__(self, roi_image, roi_x=30, roi_y=30, roi_width=25, roi_height=25):
        self.cycle = 0.5
        self.roi_image = roi_image
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.running = False
        self.detection_task = None
        self.message_bus = MessageBus()
        self.last_detection_state = None  # Track previous state (True/False/None)
        self.frames_since_state_change = 0
        self.continuous_update_interval = 10  # Publish continuous updates every N frames
        
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
        
        # Balanced confidence tracking
        self.confidence = 0.5  # Start at neutral confidence
        self.confidence_increment = 0.08  # Slightly reduced for smoother transitions
        self.confidence_decrement = 0.08  # Made equal to increment for balance
        self.max_confidence = 1.0
        self.min_confidence = 0.0
        
        print("Init detector")
        if self.roi_image:
            cv2.namedWindow('ROI Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ROI Detection', 400, 300)
            cv2.waitKey(1)

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
            # Even when no logo detected, provide meaningful confidence based on pixel count
            weak_confidence = min(0.4, colored_pixel_count / self.min_red_pixels * 0.4)
            return weak_confidence, combined_mask, colored_pixel_count
        
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

    async def _detect_loop(self, page):
        """Main detection loop with balanced confidence reporting"""
        consecutive_detections = 0
        consecutive_no_detections = 0
        
        while self.running:
            try:
                # Take screenshot
                screenshot = await page.screenshot()
                img = Image.open(io.BytesIO(screenshot))
                img_np = np.array(img)
                # Convert RGB to BGR for OpenCV
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Extract ROI with buffer
                buffer = 80
                img_height, img_width = img_np.shape[:2]
                
                crop_x1 = max(0, self.roi_x - buffer)
                crop_y1 = max(0, self.roi_y - buffer)
                crop_x2 = min(img_width, self.roi_x + self.roi_width + buffer)
                crop_y2 = min(img_height, self.roi_y + self.roi_height + buffer)
                
                # Crop the image around ROI
                cropped_img = img_np[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Adjust ROI coordinates relative to cropped image
                adj_roi_x = self.roi_x - crop_x1
                adj_roi_y = self.roi_y - crop_y1
                
                # Extract ROI from cropped image
                roi = cropped_img[adj_roi_y:adj_roi_y+self.roi_height, 
                                adj_roi_x:adj_roi_x+self.roi_width]
                
                # Detect logo
                logo_confidence, color_mask, colored_pixels = self.detect_red_logo(roi)
                logo_detected = logo_confidence > self.detection_threshold
                
                # Update running confidence with balanced smoothing
                if logo_detected:
                    self.confidence = min(self.max_confidence, 
                                        self.confidence + self.confidence_increment)
                else:
                    self.confidence = max(self.min_confidence, 
                                        self.confidence - self.confidence_decrement)
                
                # Track state changes and frame counts
                state_changed = False
                if self.last_detection_state is None:
                    # First detection
                    self.last_detection_state = logo_detected
                    state_changed = True
                    self.frames_since_state_change = 0
                elif self.last_detection_state != logo_detected:
                    # State changed
                    state_changed = True
                    self.last_detection_state = logo_detected
                    self.frames_since_state_change = 0
                else:
                    self.frames_since_state_change += 1
                
                # Use consecutive detection logic for stability
                should_publish = True
                if logo_detected:
                    consecutive_detections += 1
                    consecutive_no_detections = 0
                    if consecutive_detections >= 2:  # Require 2 consecutive detections
                        should_publish = True
                else:
                    consecutive_no_detections += 1
                    consecutive_detections = 0
                    if consecutive_no_detections >= 2:  # Require 2 consecutive non-detections
                        should_publish = True
                
                # Publish updates: on state changes OR periodic continuous updates
                if should_publish and (state_changed or 
                                     self.frames_since_state_change % self.continuous_update_interval == 0):
                    await self._publish_detection_update(
                        logo_detected, logo_confidence, colored_pixels, state_changed
                    )
                
                # Optional: Print balanced debug info
                # status = "TV" if logo_detected else "ADS"
                # print(f"--{status} (conf: {logo_confidence:.2f}, running: {self.confidence:.2f}, "
                #       f"pixels: {colored_pixels}, frames: {self.frames_since_state_change})", 
                #       end='\r', flush=True)
                
                if self.roi_image:
                    # Visualization - use cropped image
                    display_img = cropped_img.copy()
                    
                    # Draw ROI rectangle on cropped image
                    color = (0, 255, 0) if logo_detected else (0, 0, 255)
                    cv2.rectangle(display_img, 
                                (adj_roi_x, adj_roi_y),
                                (adj_roi_x + self.roi_width, adj_roi_y + self.roi_height),
                                color, 1)
                    
                    # Add balanced confidence visualization
                    # confidence_text = f"Instant: {logo_confidence:.2f} | Running: {self.confidence:.2f}"
                    # status_text = f"Status: {status} | Frames: {self.frames_since_state_change}"
                    
                    # cv2.putText(display_img, confidence_text, (10, 20), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    # cv2.putText(display_img, status_text, (10, 40), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                    cv2.imshow('ROI Detection', display_img)

                cv2.waitKey(1)
                await asyncio.sleep(self.cycle)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in detection loop: {e}")
                await asyncio.sleep(1)