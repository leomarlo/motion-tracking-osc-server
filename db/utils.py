    
    

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from db.base import Base
from db.timeDifference import TimeDifference
from config.db_config import DBCONFIG
from utils.conversion import datetimeToMicroseconds 
from datetime import datetime


def setNewTimes(session: Session) -> int :
    currentTimeEntry = session.query(TimeDifference).filter(TimeDifference.name == 'currentTime').first()
    previousTimeEntry = session.query(TimeDifference).filter(TimeDifference.name == 'previousTime').first() 
    newCurrentTime = datetimeToMicroseconds(datetime.now())
    deltaTime = newCurrentTime - currentTimeEntry.time
    previousTimeEntry.time = currentTimeEntry.time
    currentTimeEntry.time = newCurrentTime
    session.commit()
    return deltaTime
    