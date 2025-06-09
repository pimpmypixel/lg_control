import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Create logger
        self.logger = logging.getLogger('TVDetection')
        self.logger.setLevel(logging.DEBUG)
        
        # Create storage directory if it doesn't exist
        os.makedirs('storage', exist_ok=True)
        
        # Create file handler
        log_file = os.path.join('storage', 'log.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set formatters
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self._initialized = True
        self.info("Logger initialized")
    
    def set_level(self, level: str):
        """Set the logging level"""
        level = level.upper()
        if level == 'DEBUG':
            self.logger.setLevel(logging.DEBUG)
        elif level == 'INFO':
            self.logger.setLevel(logging.INFO)
        elif level == 'WARNING':
            self.logger.setLevel(logging.WARNING)
        elif level == 'ERROR':
            self.logger.setLevel(logging.ERROR)
        elif level == 'CRITICAL':
            self.logger.setLevel(logging.CRITICAL)
        else:
            self.warning(f"Invalid log level: {level}")
    
    def debug(self, message: str):
        """Log a debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log a critical message"""
        self.logger.critical(message)
    
    def get_log_file(self) -> str:
        """Get the path to the log file"""
        return os.path.join('storage', 'log.log') 