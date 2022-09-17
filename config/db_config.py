from sqlite3 import DateFromTicks

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine

class DBCONFIG:
    connectionString = 'sqlite:///data/cache.db'

    DBSession : Session
    
    def setDBSession():
        engine = create_engine(DBCONFIG.connectionString)
        DBCONFIG.DBSession = sessionmaker(bind=engine)
    
    