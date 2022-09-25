from sqlite3 import DateFromTicks

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine

class DBCONFIG:
    connectionString = 'sqlite:///data/desiremachine.db'
    DBSession : Session
    engine = create_engine(connectionString)
    STORE_RECORDINGS_IN_DB = {
        'gyro': False,
        'mediapipePose': False
    }
    
    def setDBSession():
        engine = create_engine(DBCONFIG.connectionString)
        DBCONFIG.DBSession = sessionmaker(bind=engine)
    
    def startStoringToDataBase(table=None):
        if table==None:
            for tbl in DBCONFIG.STORE_RECORDINGS_IN_DB:
                DBCONFIG.STORE_RECORDINGS_IN_DB[tbl] = True
        DBCONFIG.STORE_RECORDINGS_IN_DB[table] = True
    
    def stopStoringToDataBase(table=None):
        if table==None:
            for tbl in DBCONFIG.STORE_RECORDINGS_IN_DB:
                DBCONFIG.STORE_RECORDINGS_IN_DB[tbl] = False

        DBCONFIG.STORE_RECORDINGS_IN_DB[table] = False
    
    def isCurrentDBSessionOngoing():
        return any(DBCONFIG.STORE_RECORDINGS_IN_DB.values())
    