import cv2
import numpy as np
import time
from datetime import datetime
import threading
import queue
import json
import os
import hashlib
import uuid
from classes.utils.ntp import NTPClient
from classes.utils.db import DatabaseManager
from classes.utils.session import SessionManager

class TVColorDetector:
    def __init__(self, session_hash, camera_index=1):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # NTP client for time synchronization
        self.ntp_client = NTPClient()
        
        # Database and session management
        self.db_manager = DatabaseManager()
        self.session_manager = SessionManager(self.db_manager)
        self.session_manager.session_hash = session_hash
        
        # Color tracking
        self.avg_colors = queue.Queue(maxsize=100)  # Store last 100 color readings
        
        # TV detection confidence tracking
        self.detection_confidence = 0.0
        self.last_detected_roi = None
        self.start_time = time.time()
        self.locked_roi = None  # Store the locked ROI once confidence is 1.0
        
        # Load saved ROI
        self.load_saved_roi()
    
    def detect_tv_rectangle(self, frame):
        """Detect TV/rectangle using edge detection, contour finding, and 16:9 aspect ratio validation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with multiple thresholds for better detection
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges3 = cv2.Canny(blurred, 80, 200)
        edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
        
        # Morphological operations to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the best TV-like rectangular contour
        best_score = 0
        best_contour = None
        best_corrected_corners = None
        
        frame_center = np.array([frame.shape[1]/2, frame.shape[0]/2])
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > 15000:  # Minimum area threshold for TV
                    # Calculate the score for this contour
                    score, corrected_corners = self.evaluate_tv_candidate(approx, area, frame_center)
                    if score > best_score:
                        best_score = score
                        best_contour = approx
                        best_corrected_corners = corrected_corners
        
        return best_corrected_corners if best_corrected_corners is not None else best_contour
    
    def evaluate_tv_candidate(self, corners, area, frame_center):
        """Evaluate how well a 4-corner contour matches a 16:9 TV with perspective correction"""
        if len(corners) != 4:
            return 0, None
        
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self.order_corners(corners.reshape(4, 2))
        
        # Calculate center of the contour
        contour_center = np.mean(corners, axis=0)
        
        # Calculate distance from frame center (normalized by frame size)
        center_distance = np.linalg.norm(contour_center - frame_center) / np.linalg.norm(frame_center)
        center_score = max(0, 1 - center_distance)  # Higher score for more centered rectangles
        
        # Calculate current dimensions in image space
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])
        
        avg_width = (top_width + bottom_width) / 2
        avg_height = (left_height + right_height) / 2
        
        if avg_height == 0:
            return 0, None
        
        # Calculate aspect ratio
        aspect_ratio = avg_width / avg_height
        
        # Score based on how close to 16:9 (1.778) the aspect ratio is
        target_ratio = 16.0 / 9.0
        ratio_score = max(0, 1 - abs(aspect_ratio - target_ratio) / target_ratio)
        
        # Score based on area (larger is generally better for TV detection)
        area_score = min(1.0, area / 100000)  # Normalize to reasonable TV size
        
        # Score based on rectangle regularity (parallel sides, right angles)
        regularity_score = self.calculate_rectangle_regularity(corners)
        
        # Perspective correction score (less skew is better)
        perspective_score = self.calculate_perspective_score(corners)
        
        # Combined score with weights
        total_score = (ratio_score * 0.3 + 
                      area_score * 0.2 + 
                      regularity_score * 0.2 + 
                      perspective_score * 0.2 +
                      center_score * 0.1)  # Added center score
        
        # Apply perspective correction to get ideal 16:9 rectangle
        corrected_corners = self.correct_perspective_to_16_9(corners, avg_width, avg_height)
        
        return total_score, corrected_corners
    
    def order_corners(self, pts):
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""
        # Sort by y-coordinate
        sorted_pts = pts[np.argsort(pts[:, 1])]
        
        # Top two points
        top_pts = sorted_pts[:2]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
        
        # Bottom two points
        bottom_pts = sorted_pts[2:]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
        
        return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
    
    def calculate_rectangle_regularity(self, corners):
        """Calculate how regular/rectangular the shape is"""
        # Calculate side lengths
        sides = []
        for i in range(4):
            side_length = np.linalg.norm(corners[(i+1)%4] - corners[i])
            sides.append(side_length)
        
        # Check if opposite sides are similar (parallel sides should be equal)
        opposite_side_diff1 = abs(sides[0] - sides[2]) / max(sides[0], sides[2])
        opposite_side_diff2 = abs(sides[1] - sides[3]) / max(sides[1], sides[3])
        
        # Calculate angles (should be close to 90 degrees)
        angles = []
        for i in range(4):
            v1 = corners[i] - corners[(i-1)%4]
            v2 = corners[(i+1)%4] - corners[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(abs(90 - angle))
        
        # Score based on how close sides and angles are to perfect rectangle
        side_regularity = 1 - (opposite_side_diff1 + opposite_side_diff2) / 2
        angle_regularity = 1 - np.mean(angles) / 90
        
        return (side_regularity + angle_regularity) / 2
    
    def calculate_perspective_score(self, corners):
        """Calculate perspective distortion score (less distortion = higher score)"""
        # Calculate the degree of perspective distortion
        # In a non-distorted rectangle, parallel lines should remain parallel
        
        # Top and bottom edges
        top_edge = corners[1] - corners[0]
        bottom_edge = corners[2] - corners[3]
        
        # Left and right edges  
        left_edge = corners[3] - corners[0]
        right_edge = corners[2] - corners[1]
        
        # Calculate parallelism (dot product of normalized vectors)
        top_norm = top_edge / np.linalg.norm(top_edge)
        bottom_norm = bottom_edge / np.linalg.norm(bottom_edge)
        left_norm = left_edge / np.linalg.norm(left_edge)  
        right_norm = right_edge / np.linalg.norm(right_edge)
        
        horizontal_parallel = abs(np.dot(top_norm, bottom_norm))
        vertical_parallel = abs(np.dot(left_norm, right_norm))
        
        return (horizontal_parallel + vertical_parallel) / 2
    
    def correct_perspective_to_16_9(self, corners, width, height):
        """Apply perspective correction to create ideal 16:9 rectangle"""
        # Calculate center point
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        
        # Determine the corrected dimensions maintaining 16:9 ratio
        if width / height > 16.0 / 9.0:
            # Width is limiting factor
            corrected_width = width
            corrected_height = width * 9.0 / 16.0
        else:
            # Height is limiting factor
            corrected_height = height
            corrected_width = height * 16.0 / 9.0
        
        # Create ideal rectangle centered at the detected center
        half_width = corrected_width / 2
        half_height = corrected_height / 2
        
        ideal_corners = np.array([
            [center_x - half_width, center_y - half_height],  # top-left
            [center_x + half_width, center_y - half_height],  # top-right
            [center_x + half_width, center_y + half_height],  # bottom-right
            [center_x - half_width, center_y + half_height]   # bottom-left
        ], dtype=np.float32)
        
        # Apply weighted blend between detected corners and ideal rectangle
        # This preserves the general position while correcting aspect ratio
        blend_factor = 0.3  # How much to blend towards ideal rectangle
        corrected_corners = (1 - blend_factor) * corners + blend_factor * ideal_corners
        
        return corrected_corners.astype(np.int32)
    
    def calculate_average_color(self, frame, roi_contour=None):
        """Calculate average color in ROI with perspective correction"""
        if roi_contour is not None:
            # Use detected TV rectangle with perspective correction
            if len(roi_contour) == 4:
                # Apply perspective transformation to get undistorted view
                corrected_frame = self.apply_perspective_correction(frame, roi_contour)
                if corrected_frame is not None:
                    # Sample from the perspective-corrected frame
                    roi_pixels = corrected_frame.reshape(-1, 3)
                    # Filter out black pixels that might be from transformation padding
                    roi_pixels = roi_pixels[np.sum(roi_pixels, axis=1) > 30]
                else:
                    # Fallback to simple mask
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [roi_contour], 255)
                    roi_pixels = frame[mask > 0]
            else:
                # Fallback for non-4-corner contours
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [roi_contour], 255)
                roi_pixels = frame[mask > 0]
        else:
            return None, None
        
        if len(roi_pixels) > 0:
            avg_bgr = np.mean(roi_pixels, axis=0)
            avg_rgb = avg_bgr[::-1]  # Convert BGR to RGB
            return avg_bgr, avg_rgb
        
        return None, None
    
    def apply_perspective_correction(self, frame, corners):
        """Apply perspective transformation to correct skewed TV view"""
        try:
            # Ensure corners are properly ordered
            if len(corners) == 4:
                corners = self.order_corners(corners.reshape(4, 2))
                
                # Calculate dimensions of the corrected rectangle
                top_width = np.linalg.norm(corners[1] - corners[0])
                bottom_width = np.linalg.norm(corners[2] - corners[3])  
                left_height = np.linalg.norm(corners[3] - corners[0])
                right_height = np.linalg.norm(corners[2] - corners[1])
                
                max_width = int(max(top_width, bottom_width))
                max_height = int(max(left_height, right_height))
                
                # Ensure 16:9 aspect ratio for output
                if max_width / max_height > 16.0 / 9.0:
                    output_width = max_width
                    output_height = int(max_width * 9.0 / 16.0)
                else:
                    output_height = max_height
                    output_width = int(max_height * 16.0 / 9.0)
                
                # Define destination points for perspective correction
                dst_points = np.array([
                    [0, 0],
                    [output_width - 1, 0],
                    [output_width - 1, output_height - 1], 
                    [0, output_height - 1]
                ], dtype=np.float32)
                
                # Calculate perspective transformation matrix
                src_points = corners.astype(np.float32)
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                
                # Apply perspective transformation
                corrected = cv2.warpPerspective(frame, matrix, (output_width, output_height))
                
                return corrected
                
        except Exception as e:
            print(f"Perspective correction failed: {e}")
            
        return None
    
    def save_color_data(self, rgb_values, timestamp):
        """Save color data to database"""
        self.session_manager.save_color_data(rgb_values, timestamp)
    
    def save_roi(self, roi):
        """Save the ROI coordinates to database"""
        self.session_manager.save_roi(roi)
    
    def load_saved_roi(self):
        """Load the most recent ROI coordinates from database"""
        roi = self.session_manager.load_saved_roi()
        if roi is not None:
            self.locked_roi = roi
            self.detection_confidence = 1.0
            self.session_manager.set_recording_state(True)
        else:
            self.locked_roi = None
            self.detection_confidence = 0.0
            self.session_manager.set_recording_state(False)
    
    def update_detection_confidence(self, current_roi):
        """Update detection confidence based on stability of ROI detection"""
        # If we have a locked ROI, don't update confidence
        if self.locked_roi is not None:
            return
            
        if current_roi is None:
            self.detection_confidence = max(0, self.detection_confidence - 0.1)
            return
        
        if self.last_detected_roi is None:
            self.last_detected_roi = current_roi
            self.detection_confidence = 0.3  # Start with higher initial confidence
            return
        
        # Calculate similarity between current and last ROI
        current_center = np.mean(current_roi, axis=0)
        last_center = np.mean(self.last_detected_roi, axis=0)
        
        # Calculate center distance (normalized)
        center_distance = np.linalg.norm(current_center - last_center) / np.linalg.norm(current_center)
        
        # Calculate area similarity
        current_area = cv2.contourArea(current_roi)
        last_area = cv2.contourArea(self.last_detected_roi)
        area_ratio = min(current_area, last_area) / max(current_area, last_area)
        
        # Update confidence based on stability with faster increase
        stability_score = (1 - center_distance) * area_ratio
        self.detection_confidence = min(1.0, self.detection_confidence + stability_score * 0.2)
        
        self.last_detected_roi = current_roi
        
        # Lock ROI and start recording when confidence reaches 1.0
        if self.detection_confidence >= 1.0:
            self.locked_roi = current_roi
            self.session_manager.set_recording_state(True)
            print("ROI locked at full confidence. Starting color recording...")
            # Save the ROI coordinates
            self.save_roi(current_roi)
    
    def reset_detection(self):
        """Reset the detection system"""
        self.detection_confidence = 0.0
        self.last_detected_roi = None
        self.locked_roi = None
        self.start_time = time.time()
        self.session_manager.set_recording_state(False)
        # Remove saved ROI file if it exists
        try:
            if os.path.exists('tv_roi.json'):
                os.remove('tv_roi.json')
                print("Saved ROI file removed")
        except Exception as e:
            print(f"Error removing saved ROI file: {e}")
        print("Detection system reset")
    
    def draw_info(self, frame, avg_bgr, avg_rgb, roi_contour=None):
        """Draw information overlay on frame"""
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Scale down the frame by 50%
        height, width = display_frame.shape[:2]
        display_frame = cv2.resize(display_frame, (width//2, height//2))
        
        # Scale down ROI coordinates if present
        if roi_contour is not None:
            roi_contour = roi_contour // 2
        
        # Draw detected TV rectangle and apply dark mask outside ROI
        if roi_contour is not None:
            # Only apply dark mask if we have full confidence
            if self.detection_confidence >= 1.0:
                # Create mask for the ROI
                mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [roi_contour], 255)
                
                # Create dark overlay
                overlay = display_frame.copy()
                overlay[mask == 0] = overlay[mask == 0] * 0.3  # Darken non-ROI area
                
                # Blend the overlay with the original frame
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Draw ROI contour
            cv2.drawContours(display_frame, [roi_contour], -1, (255, 0, 0), 2)
        
        # Draw color information if recording
        # if self.session_manager.get_recording_state() and avg_bgr is not None and avg_rgb is not None:
        #     # Color swatch
        #     cv2.rectangle(display_frame, (10, 50), (100, 100), avg_bgr.tolist(), -1)
        #     cv2.rectangle(display_frame, (10, 50), (100, 100), (255, 255, 255), 2)
            
        #     # Color values
        #     bgr_text = f"BGR: ({int(avg_bgr[0])}, {int(avg_bgr[1])}, {int(avg_bgr[2])})"
        #     rgb_text = f"RGB: ({int(avg_rgb[0])}, {int(avg_rgb[1])}, {int(avg_rgb[2])})"
            
        #     cv2.putText(display_frame, bgr_text, (110, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #     cv2.putText(display_frame, rgb_text, (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def run(self):
        """Main execution loop"""
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam TV Color Detector Started")
        print("Instructions:")
        print("- System will automatically detect TV in center of frame")
        print("- Press 'r' to reset detection")
        print("- Press 's' to sync NTP time")
        print("- Press 'q' to quit")
        
        cv2.namedWindow('TV Color Detection')
        last_ntp_sync = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Periodic NTP sync (every 5 minutes)
                if time.time() - last_ntp_sync > 300:
                    threading.Thread(target=self.sync_ntp, daemon=True).start()
                    last_ntp_sync = time.time()
                
                # Use locked ROI if available, otherwise detect
                roi_contour = self.locked_roi if self.locked_roi is not None else self.detect_tv_rectangle(frame)
                
                # Update detection confidence
                self.update_detection_confidence(roi_contour)
                
                # Calculate average color if we're recording
                avg_bgr, avg_rgb = None, None
                if self.session_manager.get_recording_state() and roi_contour is not None:
                    avg_bgr, avg_rgb = self.calculate_average_color(frame, roi_contour)
                    
                    # Store color data with timestamp
                    if avg_rgb is not None:
                        # Save to database
                        self.save_color_data(avg_rgb, self.ntp_client.get_ntp_time())
                        
                        # Also keep in memory queue
                        color_data = {
                            'timestamp': self.ntp_client.get_ntp_time(),
                            'rgb': avg_rgb,
                            'bgr': avg_bgr
                        }
                        
                        if not self.avg_colors.full():
                            self.avg_colors.put(color_data)
                        else:
                            try:
                                self.avg_colors.get_nowait()
                            except queue.Empty:
                                pass
                            self.avg_colors.put(color_data)
                
                # Draw information overlay and get display frame
                display_frame = self.draw_info(frame, avg_bgr, avg_rgb, roi_contour)
                cv2.imshow('TV Color Detection', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset detection
                    self.reset_detection()
                elif key == ord('s'):
                    # Sync NTP
                    threading.Thread(target=self.ntp_client.sync_ntp, daemon=True).start()
                    print("NTP sync initiated...")
        
        finally:
            self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            # Release webcam
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                print("Webcam released")
            
            # Close all OpenCV windows
            cv2.destroyAllWindows()
            print("OpenCV windows closed")
            
            # Clear the color queue
            while not self.avg_colors.empty():
                try:
                    self.avg_colors.get_nowait()
                except queue.Empty:
                    break
            
            # Reset detection state
            self.detection_confidence = 0.0
            self.last_detected_roi = None
            self.locked_roi = None
            
            print("TV Color Detector cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Initialize the detector
    detector = TVColorDetector()
    
    try:
        # Start a new session
        detector.session_manager.start_new_session()
        print("Started new recording session")
        
        # Run the detector
        detector.run()
        
    except KeyboardInterrupt:
        print("\nStopping color detection...")
    finally:
        # End the session
        detector.session_manager.end_session()
        print("Ended recording session")
        
        # Cleanup
        detector.cleanup()