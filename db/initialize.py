

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from utils.conversion import datetimeToMicroseconds
from db.timeDifference import TimeDifference #, Universe, Base
from db.dmxUniverse import DMXUniverse
from db.session import RecordingSession
from db.base import Base
from config.db_config import DBCONFIG
from config.server_config import CONFIG
from utils.conversion import datetimeToMicroseconds 
from datetime import datetime
import os



def initializeTables():
    engine = create_engine(DBCONFIG.connectionString)
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine

    # print('schema', Base.metadata.schema)
    # print('tables', Base.metadata.tables)



def initTimedifferences(session):
    # Insert a Person in the person table
    currentTimeExists = session.query(TimeDifference).filter(TimeDifference.name=='currentTime').all()
    previousTimeExists = session.query(TimeDifference).filter(TimeDifference.name=='previousTime').all()
    if (not currentTimeExists):
        firstCurrentTime = TimeDifference(time=datetimeToMicroseconds(datetime.now()) ,name='currentTime')
        session.add(firstCurrentTime)
    if (not previousTimeExists):
        firstPreviousTime = TimeDifference(time=datetimeToMicroseconds(datetime.now()) ,name='previousTime')
        session.add(firstPreviousTime)
    if (not (currentTimeExists and previousTimeExists)):
        session.commit()
    # session.close()


def initializeUniverseToDB(session, file='DMX-universe - desire-Hallein.csv'):
    ## TODO: don't overwrite anything
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', file), header=None)
    df.drop(df[pd.isna(df[2])].index, axis=0, inplace=True)
    # engine = create_engine(DBCONFIG.connectionString)
    # df.to_sql(Universe.__tablename__, engine, if_exists="replace")

    for i, row in df.iterrows():
        exists = session.query(DMXUniverse.id).filter_by(address=row[2]).first() is not None
        if exists:
            continue
        channel = DMXUniverse(reference=int(row[1]), address=row[2])
        session.add(channel)
    session.commit()
    # session.close()


def rememberThisRecordingSession(session):
    recordingSessionData = dict(
        starting_timestamp_in_microsecs=datetimeToMicroseconds(datetime.now()),
        ip_address=CONFIG.myIP,
        port=CONFIG.serverPort,
        host_name=CONFIG.myHostName
    )
    print(recordingSessionData)
    rs = RecordingSession(**recordingSessionData)
    session.add(rs)
    session.commit()


# allEntries = session.query(Universe).all()
# print(allEntries)
# for entry in allEntries:
#     print(entry.address)
    