from sqlalchemy import Column, ForeignKey, Integer, String, Float
from db.base import Base

class MediapipePose(Base):
    __tablename__ = 'mediapipePose'
    id = Column(Integer, primary_key=True)
    sessionid = Column(Integer, nullable=False)
    frameid = Column(Integer, nullable=False)
    timestamp_in_microsecs = Column(Integer, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    z = Column(Integer, nullable=False)
    landmarkid = Column(Integer, nullable=False)
    landmarkname  = Column(String(250), nullable=False)



