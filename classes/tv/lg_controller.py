import asyncio
import json
import os
from aiowebostv import WebOsClient

class LGTVController:
    def __init__(self, ip_address="192.168.1.218", storage_path='./tv_client_key.json'):
        self.ip_address = ip_address
        self.client = None
        self.key_file = storage_path
        print('Init controller')

    def _load_client_key(self):
        """Load saved client key if available"""
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'r') as f:
                    data = json.load(f)
                    if data.get('ip') == self.ip_address:
                        print("Client key loaded")
                        return data.get('key')
        except Exception as e:
            print(f"Could not load client key: {e}")
        return None

    def _save_client_key(self, key):
        """Save client key for future use"""
        try:
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            with open(self.key_file, 'w') as f:
                json.dump({'ip': self.ip_address, 'key': key}, f)
            print("Client key saved")
        except Exception as e:
            print(f"Could not save client key: {e}")

    async def connect(self):
        """Connect to the LG TV"""
        if self.client:
            print("Already connected")
            return True
            
        try:
            client_key = self._load_client_key()
            
            # Create WebOsClient with aiowebostv
            self.client = WebOsClient(self.ip_address, client_key=client_key)
            await self.client.connect()
            
            # Save the client key if we got a new one
            if self.client.client_key != client_key:
                self._save_client_key(self.client.client_key)
                
            print(f"Connected to LG TV at {self.ip_address}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to TV: {e}")
            self.client = None
            return False

    async def disconnect(self):
        """Disconnect from the LG TV"""
        if self.client:
            try:
                await self.client.disconnect()
                print("Disconnected from TV")
            except Exception as e:
                print(f"Error during disconnect: {e}")
            finally:
                self.client = None

    async def _ensure_connected(self):
        """Ensure we have a valid connection"""
        if not self.client:
            success = await self.connect()
            if not success:
                raise ConnectionError("Could not establish connection to TV")

    async def get_mute_status(self):
        """Get current mute status"""
        await self._ensure_connected()
        
        try:
            return await self.client.get_muted()
        except Exception as e:
            print(f"Failed to get mute status: {e}")
            return None

    async def set_mute(self, mute=True):
        """Set mute status"""
        await self._ensure_connected()
        
        try:
            await self.client.set_mute(mute)
            # print(f"TV {'muted' if mute else 'unmuted'}")
            return True
        except Exception as e:
            print(f"Failed to set mute to {mute}: {e}")
            return False

    async def get_current_app(self):
        """Get the current running app on the TV"""
        await self._ensure_connected()
        
        try:
            app_info = await self.client.get_current_app()
            # print(f"Current app: {app_info}")
            return app_info
        except Exception as e:
            print(f"Failed to get current app: {e}")
            return None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


# # Example usage
# async def main():
#     tv_ip = "192.168.1.218"
    
#     print("LG TV Controller")
#     print("===============")
    
#     try:
#         async with LGTVController(tv_ip) as tv:
#             # Get current app
#             print("\n=== Current App ===")
#             current_app = await tv.get_current_app()
#             if current_app is not None:
#                 print(f"Currently running: {current_app}")

#             # Get current mute status
#             print("\n=== Current Mute Status ===")
#             current_mute = await tv.get_mute_status()
#             if current_mute is not None:
#                 print(f"Currently {'muted' if current_mute else 'unmuted'}")
                
#                 # Toggle mute
#                 new_mute = not current_mute
#                 success = await tv.set_mute(new_mute)
#                 if success:
#                     print(f"Successfully {'muted' if new_mute else 'unmuted'} TV")
#                     # Wait a moment then toggle back
#                     await asyncio.sleep(2)
#                     await tv.set_mute(current_mute)
#                     print(f"Restored original mute state")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\nProgram terminated by user")
#     except Exception as e:
#         print(f"An error occurred: {e}") 