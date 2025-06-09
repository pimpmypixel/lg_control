import asyncio
import argparse
from classes.clients.tv2_client import Client
from classes.utils.message_bus import MessageBus, MessageType
from classes.tv.lg_controller import LGTVController

def parse_args():
    parser = argparse.ArgumentParser(description='TV2 Client with configurable options')
    parser.add_argument('--debug','-D', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--no-headless','-H', action='store_true', default=False, help='Run in headful mode')
    parser.add_argument('--no-tv','-T', action='store_true', default=False, help='No TV')
    parser.add_argument('--roi','-R', action='store_true', default=False, help='Enable ROI image')
    return parser.parse_args()

app = "tv2play"
tv_ip="192.168.1.218"
storage_path='./storage/tv_client_key.json'

async def handle_messages(tv, args):
    message_bus = MessageBus()
    current_app = 'No TV'
    if tv is not None:
        current_app = await tv.get_current_app()
    async for message in message_bus.subscribe():
        if message.type == MessageType.LOGO_DETECTED:
            mute = False
            detection = 'TV'
        elif message.type == MessageType.LOGO_LOST:
            mute = True
            detection = 'ADS'

        if not args.no_tv or current_app == app:
            success = await tv.set_mute(mute = mute)
            # status = f"-- {detection} -- Confidence: {message.data['confidence']:.2f}  -- Muted: {mute}"
        # else:
        #     status = f"\rCurrent app: {current_app}"
        # print(status, end='\r', flush=True)

async def main():
    args = parse_args()
    if args.debug is True:
        print('Debugging mode')
    try:
        client = Client()
        tv = None
        if args.no_tv is False:
            tv = LGTVController(ip_address = tv_ip, storage_path = storage_path)
            await tv.connect()
        message_handler = asyncio.create_task(handle_messages(tv, args))
        
        current_app = 'No TV'
        if tv is not None:
            current_app = await tv.get_current_app()
            
        if args.no_tv or current_app == app:
            print(f"Currently running: {current_app}")
            await client.initialize(app, args)
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
