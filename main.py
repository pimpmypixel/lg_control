import asyncio
from classes.client import Client
from classes.utils.message_bus import MessageBus, MessageType
from classes.tv.lg_controller import LGTVController

headless = True
roi_image= False
app = "tv2play"

async def handle_messages(controller = None):
    message_bus = MessageBus()
    current_app = await controller.get_current_app()
    async for message in message_bus.subscribe():
        if message.type == MessageType.LOGO_DETECTED:
            mute = False
            detection = 'TV'
        elif message.type == MessageType.LOGO_LOST:
            mute = True
            detection = 'ADS'
        if current_app == app:
            success = await controller.set_mute(mute=mute)
        message = f"-- {detection} -- Confidence: {message.data['confidence']:.2f}  -- Muted: {mute}"
        print(message, end='\r', flush=True)
        


async def main():
    try:
        client = Client()
        controller = LGTVController(ip_address="192.168.1.218",storage_path='./storage/tv_client_key.json')
        message_handler = asyncio.create_task(handle_messages(controller))
        
        async with controller:
            current_app = await controller.get_current_app()
            if current_app == app:
                print(f"Currently running: {current_app}")
                await client.initialize(headless, roi_image)
                await client.run()
            else:
                print(f"Expected app {app} but found {current_app}")
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
        if client:
            await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
