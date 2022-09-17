from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer, AsyncIOOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from config.server_config import CONFIG
from config.db_config import DBCONFIG
from sqlalchemy import create_engine
from sqlalchemy import update
from sqlalchemy.orm import sessionmaker
from db.base import Base
from db.tables import Time
from osc.handling import MediaPipe
import asyncio
from utils.conversion import datetimeToMicroseconds 
from db.initialize import initTimedifferences, initializeTables
from datetime import datetime, timedelta
from db.utils import setNewTimes


# CONFIG.loadMyIP()
## ONLY FOR TESTING. DONT CALL THIS FUNCTION OTHERWISE
# CONFIG.onlyReceive()
# CONFIG.localTesting()
DBCONFIG.setDBSession()
initializeTables()
initTimedifferences(DBCONFIG.DBSession)

if CONFIG.verbosity>1:
    print(CONFIG.myHostName, CONFIG.myIP, CONFIG.ONLY_RECEIVE)


client = SimpleUDPClient(CONFIG.clientIP, CONFIG.clientPort) 


def start_capture(address, *args):
    MediaPipe.startCapture(0)
    MediaPipe.handleCapture()

def stop_capture(address, *args):
    print( MediaPipe.cap.isOpen())
    MediaPipe.stopCapture()
    print( MediaPipe.cap.isOpen())
    MediaPipe.cap.release()
    print( MediaPipe.cap.isOpen())


def stop_server(address, *args):
    CONFIG.stopServer()


def default_handler(address, *args):
    # print(f"DEFAULT {address}: {args}")
    
    if args[0] == 'tilts':
        deltaTime = setNewTimes(DBCONFIG.DBSession)
        # print('Time in microseconds {t}'.format(t=datetimeToMicroseconds(datetime.now())))

        print("the delta time in microsecond is {t}".format(t=deltaTime))

        # print(times)
        # times.lastTime = times.currentTime
    # if CONFIG.ONLY_RECEIVE:
    #     return
    # if args[0] == 'tilts':
    #     client.send_message(address, (args[1], args[2])) 


dispatcher = Dispatcher()
dispatcher.map("/startCapture", start_capture)
dispatcher.map("/stopCapture", stop_capture)
dispatcher.map("/stopServer", stop_server)

dispatcher.set_default_handler(default_handler)
# server = BlockingOSCUDPServer((CONFIG.myIP, CONFIG.serverPort), dispatcher)
# server.serve_forever()  # Blocks forever



async def loop():
    """When this loop stops, the server will stop too"""
    while not CONFIG.STOP_SERVER:
        print("hello!")
        await asyncio.sleep(3)

async def init_main():
    server = AsyncIOOSCUDPServer((CONFIG.myIP, CONFIG.serverPort), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    await loop()  # Enter main loop of program

    transport.close()  # Clean up serve endpoint


asyncio.run(init_main())