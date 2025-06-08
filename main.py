import asyncio
from classes.clients.tv2_client import Client
from classes.utils.message_bus import MessageBus, MessageType
from classes.tv.lg_controller import LGTVController

headless = True
roi_image= False
debug_mode= True
app = "tv2play"
tv_ip="192.168.1.218"
storage_path='./storage/tv_client_key.json'

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

        if debug_mode or current_app == app:
            # success = await controller.set_mute(mute=mute)
            message = f"-- {detection} -- Confidence: {message.data['confidence']:.2f}  -- Muted: {mute}"
        else:
            message = f"Current app: {current_app}"
        print(message, end='\r', flush=True)

async def main():
    try:
        client = Client()
        controller = LGTVController(ip_address=tv_ip,storage_path=storage_path)
        await controller.connect()
        message_handler = asyncio.create_task(handle_messages(controller))
        
        async with controller:
            current_app = await controller.get_current_app()
            if debug_mode or current_app == app:
                print(f"Currently running: {current_app}")
                await client.initialize(app, headless, roi_image)
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
