from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from config.server_config import CONFIG
from config.db_config import DBCONFIG
from sqlalchemy import create_engine
from sqlalchemy import update
from sqlalchemy.orm import sessionmaker
from db.base import Base
from db.tables import Time

engine = create_engine(DBCONFIG.connectionString)
Base.metadata.create_all(engine)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
# A DBSession() instance establishes all conversations with the database
# and represents a "staging zone" for all the objects loaded into the
# database session object. Any change made against the objects in the
# session won't be persisted into the database until you call
# session.commit(). If you're not happy about the changes, you can
# revert all of them back to the last commit by calling
# session.rollback()

def datetimeToMicroseconds(dt):
    return round(dt.timestamp() * 1000000)

# Insert a Person in the person table
session = DBSession()
from datetime import datetime
currentTimeExists = session.query(Time).filter(Time.name=='currentTime').all()
previousTimeExists = session.query(Time).filter(Time.name=='previousTime').all()
if (not currentTimeExists):
    firstCurrentTime = Time(time=datetimeToMicroseconds(datetime.now()) ,name='currentTime')
    session.add(firstCurrentTime)
if (not previousTimeExists):
    firstPreviousTime = Time(time=datetimeToMicroseconds(datetime.now()) ,name='previousTime')
    session.add(firstPreviousTime)
if (not (currentTimeExists and previousTimeExists)):
    session.commit()
session.close()



CONFIG.loadMyIP()
## ONLY FOR TESTING. DONT CALL THIS FUNCTION OTHERWISE
# CONFIG.onlyReceive()
# CONFIG.localTesting()
print(CONFIG.myHostName, CONFIG.myIP, CONFIG.ONLY_RECEIVE)
client = SimpleUDPClient(CONFIG.clientIP, CONFIG.clientPort) 
from datetime import datetime, timedelta
times = dict(lastTime = datetime.now(), currentTime = datetime.now()) 

def print_handler(address, *args):
    print(f"{address}: {args}")
    client.send_message("/some/address", 123) 


def default_handler(address, *args):
    # print(f"DEFAULT {address}: {args}")
    
    if args[0] == 'tilts':
        session = DBSession()
        currentTimeEntry = session.query(Time).filter(Time.name == 'currentTime').first()
        previousTimeEntry = session.query(Time).filter(Time.name == 'previousTime').first() 
        newCurrentTime = datetimeToMicroseconds(datetime.now())
        deltaTime = newCurrentTime - currentTimeEntry.time
        previousTimeEntry.time = currentTimeEntry.time
        currentTimeEntry.time = newCurrentTime
        session.commit()
        session.close()
        # print('Time in microseconds {t}'.format(t=datetimeToMicroseconds(datetime.now())))

        print("the delta time in microsecond is {t}".format(t=deltaTime))

        # print(times)
        # times.lastTime = times.currentTime
    # if CONFIG.ONLY_RECEIVE:
    #     return
    # if args[0] == 'tilts':
    #     client.send_message(address, (args[1], args[2])) 


dispatcher = Dispatcher()
dispatcher.map("/something/*", print_handler)
dispatcher.set_default_handler(default_handler)



server = BlockingOSCUDPServer((CONFIG.myIP, CONFIG.serverPort), dispatcher)
server.serve_forever()  # Blocks forever