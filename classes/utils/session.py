import hashlib
import uuid
from datetime import datetime
from classes.utils.db import DatabaseManager

class SessionManager:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or DatabaseManager()
        self.session_hash = None
        self.is_recording = False
    
    def start_new_session(self):
        """Start a new recording session"""
        self.session_hash = self.db_manager.start_new_session()
        return self.session_hash
    
    def end_session(self):
        """End the current recording session"""
        if self.session_hash:
            self.db_manager.end_session(self.session_hash)
            self.session_hash = None
            self.is_recording = False
    
    def get_session_hash(self):
        """Get the current session hash"""
        return self.session_hash
    
    def is_active(self):
        """Check if there is an active session"""
        return self.session_hash is not None
    
    def set_recording_state(self, state):
        """Set the recording state"""
        self.is_recording = state
    
    def get_recording_state(self):
        """Get the current recording state"""
        return self.is_recording
    
    def save_color_data(self, rgb_values, timestamp):
        """Save color data to database"""
        if self.session_hash and self.is_recording:
            self.db_manager.save_color_data(self.session_hash, rgb_values, timestamp)
    
    def save_roi(self, roi):
        """Save the ROI coordinates to database"""
        if roi is not None and len(roi) == 4 and self.session_hash:
            self.db_manager.save_roi(self.session_hash, roi)
    
    def load_saved_roi(self):
        """Load the most recent ROI coordinates from database"""
        return self.db_manager.load_saved_roi() 