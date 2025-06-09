import os
import sys
import asyncio
import argparse
from classes.utils.db import DatabaseManager
from classes.utils.session import SessionManager
from classes.webcam.TVColorDetection import TVColorDetector
from classes.detect_logo.tv2_play_logo import TV2PlayLogoDetector
from classes.playwright.tv2_client import TV2Client
from classes.tv.lg_controller import LGTVController

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Smart TV Streaming Controller')
    parser.add_argument('--no-headless', action='store_true', help='Run browser in non-headless mode')
    parser.add_argument('--roi', action='store_true', help='Enable ROI image display')
    parser.add_argument('--webcam', action='store_true', help='Enable TV color detection using webcam')
    return parser.parse_args()

def setup_database():
    """Initialize the database and create necessary tables"""
    db_manager = DatabaseManager()
    return db_manager

def start_session(db_manager):
    """Start a new recording session"""
    session_manager = SessionManager(db_manager)
    session_hash = session_manager.start_new_session()
    return session_manager, session_hash

async def run_tv_controller(args):
    """Run the LG TV controller"""
    tv_controller = LGTVController(ip_address="192.168.1.218", storage_path='./storage/tv_client_key.json')
    await tv_controller.connect()
    return tv_controller

async def main():
    args = parse_args()
    
    try:
        db_manager = setup_database()
        session_manager, session_hash = start_session(db_manager)
        tv_controller = await run_tv_controller(args)
        browser = TV2Client()
        logo_detector = TV2PlayLogoDetector(roi_image=args.roi, roi_x=30, roi_y=10, roi_width=25, roi_height=25)

        webcam_color_detector = None
        if args.webcam:
            print("Initializing TV color detector...")
            webcam_color_detector = TVColorDetector(session_hash)

        await browser.initialize("tv2play", args) 
        # browser_task = asyncio.create_task(start_browser(session_hash, args))
        
        # Run the detectors
        try:
            if webcam_color_detector:
                await webcam_color_detector.run()

            page = await browser._start_stream()
            await logo_detector.start_detection(page)
            await browser.run()

        except KeyboardInterrupt:
            print("\nStopping color detection...")
        finally:
            session_manager.end_session()
            # Cleanup
            if webcam_color_detector:
                await webcam_color_detector.cleanup()
                
            await logo_detector.stop_detection()
            await browser.cleanup()
            await tv_controller.disconnect()
            
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())