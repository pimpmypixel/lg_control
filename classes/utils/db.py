import sqlite3
import os
from datetime import datetime
import hashlib
import uuid

class DatabaseManager:
    def __init__(self, storage_dir='../../storage', db_name='tv_colors.db'):
        self.storage_dir = storage_dir
        self.db_path = os.path.join(storage_dir, db_name)
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database and create tables if they don't exist"""
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.storage_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_hash TEXT PRIMARY KEY,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP
                )
            ''')
            
            # Create colorcodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS colorcodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_hash TEXT,
                    r INTEGER,
                    g INTEGER,
                    b INTEGER,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_hash) REFERENCES sessions(session_hash)
                )
            ''')
            
            # Create ROI table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS roi_coordinates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_hash TEXT,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    x3 INTEGER,
                    y3 INTEGER,
                    x4 INTEGER,
                    y4 INTEGER,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_hash) REFERENCES sessions(session_hash)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("Database setup completed")
        except Exception as e:
            print(f"Error setting up database: {e}")
    
    def start_new_session(self):
        """Start a new recording session and return the session hash"""
        try:
            # Generate a unique 8-character session hash
            session_hash = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:8]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert new session
            cursor.execute('''
                INSERT INTO sessions (session_hash, start_date)
                VALUES (?, ?)
            ''', (session_hash, datetime.now()))
            
            conn.commit()
            conn.close()
            print(f"New session started: {session_hash}")
            return session_hash
        except Exception as e:
            print(f"Error starting new session: {e}")
            return None
    
    def end_session(self, session_hash):
        """End the current recording session"""
        if session_hash:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Update session end date
                cursor.execute('''
                    UPDATE sessions
                    SET end_date = ?
                    WHERE session_hash = ?
                ''', (datetime.now(), session_hash))
                
                conn.commit()
                conn.close()
                print(f"Session ended: {session_hash}")
            except Exception as e:
                print(f"Error ending session: {e}")
    
    def save_color_data(self, session_hash, rgb_values, timestamp):
        """Save color data to database"""
        if session_hash:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Insert color data
                cursor.execute('''
                    INSERT INTO colorcodes (session_hash, r, g, b, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_hash, int(rgb_values[0]), int(rgb_values[1]), 
                      int(rgb_values[2]), timestamp))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error saving color data: {e}")
    
    def save_roi(self, session_hash, roi):
        """Save the ROI coordinates to database"""
        if roi is not None and len(roi) == 4 and session_hash:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Insert ROI coordinates
                cursor.execute('''
                    INSERT INTO roi_coordinates 
                    (session_hash, x1, y1, x2, y2, x3, y3, x4, y4, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_hash, 
                      int(roi[0][0]), int(roi[0][1]),
                      int(roi[1][0]), int(roi[1][1]),
                      int(roi[2][0]), int(roi[2][1]),
                      int(roi[3][0]), int(roi[3][1]),
                      datetime.now()))
                
                conn.commit()
                conn.close()
                print("ROI coordinates saved to database")
            except Exception as e:
                print(f"Error saving ROI coordinates: {e}")
    
    def load_saved_roi(self):
        """Load the most recent ROI coordinates from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the most recent ROI coordinates
            cursor.execute('''
                SELECT x1, y1, x2, y2, x3, y3, x4, y4
                FROM roi_coordinates
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # Convert coordinates to numpy array
                import numpy as np
                roi = np.array([
                    [result[0], result[1]],
                    [result[2], result[3]],
                    [result[4], result[5]],
                    [result[6], result[7]]
                ], dtype=np.int32)
                
                print("Loaded saved ROI from database")
                return roi
            
            return None
        except Exception as e:
            print(f"Error loading saved ROI: {e}")
            return None 