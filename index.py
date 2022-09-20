from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer, AsyncIOOSCUDPServer, ThreadingOSCUDPServer
from config.server_config import CONFIG
from config.db_config import DBCONFIG
# from sqlalchemy import create_engine
# from sqlalchemy import update
# from sqlalchemy.orm import sessionmaker
# from db.base import Base
# from db.timeDifference import TimeDifference
# from db.dmxUniverse import DMXUniverse
from utils.mediapipe import MediaPipe
from osc.client import Client
import pandas as pd
import asyncio
# from utils.conversion import datetimeToMicroseconds 
from db.initialize import initTimedifferences, initializeTables, initializeUniverseToDB
# from datetime import datetime, timedelta
from db.utils import setNewTimes


CONFIG.loadMyIP()
## ONLY FOR TESTING. DONT CALL THIS FUNCTION OTHERWISE
# CONFIG.onlyReceive()
# CONFIG.localTesting()
DBCONFIG.setDBSession()
initializeTables()
session = DBCONFIG.DBSession()
initTimedifferences(session)
initializeUniverseToDB(session)
session.close()

Client.setClient(CONFIG.clientIP, CONFIG.clientPort)

if CONFIG.verbosity>1:
    print(CONFIG.myHostName, CONFIG.myIP, CONFIG.ONLY_RECEIVE)




def print_client_port_and_address(address, *args):
    print("The ip address is {adr} and the port is {port}".format(adr=Client.client._address, port=Client.client._port))

def change_client(address, *args):
    print(args[0], args[1])
    Client.setClient(args[0], args[1])
    # Client.client = SimpleUDPClient(CONFIG.clientIP, CONFIG.clientPort) 

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
dispatcher.map("/changeClient", change_client)
dispatcher.map("/printClientInfo", print_client_port_and_address)
# dispatcher.map("/velocity", )

dispatcher.set_default_handler(default_handler)
server = ThreadingOSCUDPServer((CONFIG.myIP, CONFIG.serverPort), dispatcher)
server.serve_forever()  # Blocks forever



# async def loop():
#     """When this loop stops, the server will stop too"""
#     while not CONFIG.STOP_SERVER:
#         print("hello!")
#         await asyncio.sleep(3)

# async def init_main():
#     server = AsyncIOOSCUDPServer((CONFIG.myIP, CONFIG.serverPort), dispatcher, asyncio.get_event_loop())
#     transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

#     await loop()  # Enter main loop of program

#     transport.close()  # Clean up serve endpoint


# asyncio.run(init_main())

# df = pd.read_csv('data/DMX-universe - desire-Hallein.csv', header=None)
# df = df.drop(df[pd.isna(df[2])].index, axis=0, inplace=False)
# engine = create_engine(DBCONFIG.connectionString)
# df.to_sql(Universe.__tablename__, engine, if_exists="replace")


# for i, row in df.iterrows():
#     # values = row.todict().values()
#     # print(row[2])
#     channel = Universe(reference=int(row[1]), address=row[2])
#     session.add(channel)
# session.commit()
# session.close()

# allEntries = session.query(DMXUniverse).all()
# print(allEntries)
# for entry in allEntries:
#     print(entry.address)
    




