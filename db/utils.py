    
    

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from db.base import Base
from db.tables import Time
from config.db_config import DBCONFIG
from utils.conversion import datetimeToMicroseconds 
from datetime import datetime


def setNewTimes(DBSession: Session) -> int :
    session = DBSession()
    currentTimeEntry = session.query(Time).filter(Time.name == 'currentTime').first()
    previousTimeEntry = session.query(Time).filter(Time.name == 'previousTime').first() 
    newCurrentTime = datetimeToMicroseconds(datetime.now())
    deltaTime = newCurrentTime - currentTimeEntry.time
    previousTimeEntry.time = currentTimeEntry.time
    currentTimeEntry.time = newCurrentTime
    session.commit()
    session.close()
    return deltaTime