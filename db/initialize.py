

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.base import Base
from .tables import Time
from config.db_config import DBCONFIG
from utils.conversion import datetimeToMicroseconds 
from datetime import datetime


def initializeTables():
    engine = create_engine(DBCONFIG.connectionString)
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine


def initTimedifferences(DBSession):
    # Insert a Person in the person table
    session = DBSession()
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