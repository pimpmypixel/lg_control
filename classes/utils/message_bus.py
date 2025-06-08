import asyncio
from enum import Enum

class MessageType(Enum):
    LOGO_DETECTED = "logo_detected"
    LOGO_LOST = "logo_lost"
    ERROR = "error"

class Message:
    def __init__(self, type: MessageType, data: dict = None):
        self.type = type
        self.data = data or {}

class MessageBus:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MessageBus, cls).__new__(cls)
            cls._instance.queue = asyncio.Queue()
        return cls._instance
    
    async def publish(self, message: Message):
        await self.queue.put(message)
    
    async def subscribe(self):
        while True:
            message = await self.queue.get()
            yield message
            self.queue.task_done() 