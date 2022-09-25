from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer, AsyncIOOSCUDPServer, ThreadingOSCUDPServer
from config.server_config import CONFIG
from config.db_config import DBCONFIG
from utils.math import euclidean
from osc.client import Client
# from sqlalchemy import create_engine
# from sqlalchemy import update
# from sqlalchemy.orm import sessionmaker
# from db.base import Base
# from db.timeDifference import TimeDifference
# from db.dmxUniverse import DMXUniverse
from utils.mediapipe import MediaPipe
# from utils.mobile import calculateMobileEnergy
from osc.client import Client
import pandas as pd
import numpy as np
import asyncio
# from utils.conversion import datetimeToMicroseconds 
from db.initialize import (
    initTimedifferences,
    initializeTables,
    initializeUniverseToDB, 
    rememberThisRecordingSession)
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

def start_video_capture(address, *args):
    if args[0]==1:
        alreadyRunningRecording = DBCONFIG.isCurrentDBSessionOngoing()
        DBCONFIG.startStoringToDataBase('mediapipePose')
        if not alreadyRunningRecording:
            session = DBCONFIG.setDBSession()
            rememberThisRecordingSession(session)
            session.close()
    MediaPipe.startCapture(0)
    MediaPipe.handleCapture(with_drawing_landmarks=(args[1]==1))

def stop_video_capture(address, *args):
    MediaPipe.stopCapture()
    DBCONFIG.stopStoringToDataBase('mediapipePose')

def start_capture(address, *args):
    start_mobile_capture(address,args)
    start_video_capture(address,args)

def stop_capture(address, *args):
    stop_mobile_capture(address,args)
    stop_video_capture(address,args)

def stop_server(address, *args):
    CONFIG.stopServer()

def start_mobile_capture(address, *args):
    # print('mobile capture')
    if args[0]==1:
        # print('db storage')
        alreadyRunningRecording = DBCONFIG.isCurrentDBSessionOngoing()
        DBCONFIG.startStoringToDataBase('gyro')
        if not alreadyRunningRecording:
            
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            # engine = create_engine(DBCONFIG.connectionString)
            DBSession = sessionmaker(bind=DBCONFIG.engine)
            session = DBSession()
            print('session',session)
            rememberThisRecordingSession(session)
            session.close()
    CONFIG.startRecordingMobile()

def stop_mobile_capture(address, *args):
    CONFIG.stopRecordingMobile()
    DBCONFIG.stopStoringToDataBase('gyro')

def start_writing_to_db(address, *args):
    DBCONFIG.startStoringToDataBase()

def stop_writing_to_db(address, *args):
    DBCONFIG.stopStoringToDataBase()

def handle_mobile_data(address, *args):
    if args[0] == 'gyro':
        # print(args)
        # gyrationAlongVerticalAxis = args[3]
        # gyrationLongAxis = args[2]
        # gyrationShortAxis = args[1]
        MediaPipe.computeAllGyroEnergies(args)

def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")
    
    # if args[0] == 'tilts':
    #     deltaTime = setNewTimes(DBCONFIG.DBSession)
    #     # print('Time in microseconds {t}'.format(t=datetimeToMicroseconds(datetime.now())))

    #     print("the delta time in microsecond is {t}".format(t=deltaTime))

    #     # print(times)
    #     # times.lastTime = times.currentTime
    # # if CONFIG.ONLY_RECEIVE:
    # #     return
    # # if args[0] == 'tilts':
    # #     client.send_message(address, (args[1], args[2])) 


dispatcher = Dispatcher()
dispatcher.map("/valueToNetwork", handle_mobile_data)
dispatcher.map("/startVideoCapture", start_video_capture)
dispatcher.map("/stopVideoCapture", stop_video_capture)
dispatcher.map("/startCapture", start_capture)
dispatcher.map("/stopCapture", stop_capture)
dispatcher.map("/startMobileCapture", start_mobile_capture)
dispatcher.map("/stopMobileCapture", stop_mobile_capture)
dispatcher.map("/stopServer", stop_server)
dispatcher.map("/changeClient", change_client)
dispatcher.map("/printClientInfo", print_client_port_and_address)
dispatcher.map("/startDbWrting", start_writing_to_db)
dispatcher.map("/stopDbWrting", stop_writing_to_db)
# dispatcher.map("/velocity", )

dispatcher.set_default_handler(default_handler)
server = ThreadingOSCUDPServer((CONFIG.myIP, CONFIG.serverPort), dispatcher)
server.serve_forever()  # Blocks forever


async def loop():
    """When this loop stops, the server will stop too"""
    while not CONFIG.STOP_SERVER:
        print("hello!")
        await asyncio.sleep(3)

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
    




