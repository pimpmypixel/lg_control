import asyncio
from ..utils.logging import Logger
from ..utils.message_bus import MessageBus, MessageType
from .lg_controller import LGTVController

class TV2Client:
    def __init__(self, unified_display=None):
        self.logger = Logger()
        self.unified_display = unified_display
        self.message_bus = MessageBus()
        self.running = False
        self.tv = None
        self.current_app = None
        self.expected_app = "tv2play"
        self.tv_ip = "192.168.1.218"
        self.storage_path = './storage/tv_client_key.json'
    
    async def initialize(self):
        """Initialize the TV client"""
        try:
            # Initialize TV controller
            self.tv = LGTVController(ip_address=self.tv_ip, storage_path=self.storage_path)
            await self.tv.connect()
            self.logger.info("Connected to LG TV")
            
            # Get current app
            self.current_app = await self.tv.get_current_app()
            self.logger.info(f"Current TV app: {self.current_app}")
            
            # Update status in unified display
            if self.unified_display:
                self.unified_display.update_status({
                    'tv_control': {
                        'connected': True,
                        'current_app': self.current_app,
                        'expected_app': self.expected_app
                    }
                })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing TV client: {e}")
            if self.unified_display:
                self.unified_display.update_status({
                    'tv_control': {
                        'connected': False,
                        'error': str(e)
                    }
                })
            return False
    
    async def start(self):
        """Start the TV client"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting TV client")
        
        # Subscribe to message bus
        async for message in self.message_bus.subscribe():
            if not self.running:
                break
                
            try:
                if message.type == MessageType.LOGO_DETECTED:
                    # Logo detected - unmute TV
                    if self.tv and self.current_app == self.expected_app:
                        await self.tv.set_mute(mute=False)
                        self.logger.info("TV unmuted - logo detected")
                        
                elif message.type == MessageType.LOGO_LOST:
                    # Logo lost - mute TV
                    if self.tv and self.current_app == self.expected_app:
                        await self.tv.set_mute(mute=True)
                        self.logger.info("TV muted - logo lost")
                
                # Update status in unified display
                if self.unified_display:
                    self.unified_display.update_status({
                        'tv_control': {
                            'connected': True,
                            'current_app': self.current_app,
                            'expected_app': self.expected_app,
                            'muted': message.type == MessageType.LOGO_LOST
                        }
                    })
                
            except Exception as e:
                self.logger.error(f"Error handling message: {e}")
    
    def stop(self):
        """Stop the TV client"""
        self.running = False
        self.logger.info("TV client stopped")
    
    async def cleanup(self):
        """Clean up resources"""
        self.stop()
        if self.tv:
            await self.tv.disconnect()
            self.logger.info("TV disconnected") 